import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import json
import optuna
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold, GridSearchCV


# ===== Data splitting =====

def split_data_to_train_val_test(csv_path, ratios, output_dir, filename_prefix, filename_ending=''):
    """
    Splits the indexes of a csv containing the dataset into Train, Valdidation and Test sets.
    Has a constant seed for reproducible results.

    Output schema: {filename_prefix}_index_train{filename_ending}.csv
                   {filename_prefix}_index_validation{filename_ending}.csv
                   {filename_prefix}_index_test{filename_ending}.csv
    
    Args:
        csv_path (str): Path to csv of dataframe.
        ratios (List[float]): List of fractions for the split ratios: Train, validation, test. e.g., [0.8, 0.1, 0.1].
        output_dir (str): Path to the directory to save the index files.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        filename_ending (str, optional): The output files suffix.
    """
    df = pd.read_csv(csv_path)

    train_ratio = ratios[0]
    val_ratio = ratios[1]

    num_molecules = len(df)
    print(f"Dataset loaded with {num_molecules} samples.")

    index = np.arange(num_molecules)

    np.random.seed(42) # keep a constant seed for same results
    np.random.shuffle(index) # shuffle the index numbers

    train_split = int(np.round(train_ratio * num_molecules))
    val_split = int(np.round(val_ratio * num_molecules))

    index_train = index[:train_split]
    index_val = index[train_split : train_split + val_split]
    index_test = index[train_split + val_split :]

    # save into a csv
    os.makedirs(output_dir, exist_ok=True)
    train_path = output_dir / f'{filename_prefix}_index_train{filename_ending}.csv'
    val_path = output_dir / f'{filename_prefix}_index_validation{filename_ending}.csv'
    test_path = output_dir / f'{filename_prefix}_index_test{filename_ending}.csv'

    pd.DataFrame(index_train, columns=['index']).to_csv(train_path, index=False)
    pd.DataFrame(index_val, columns=['index']).to_csv(val_path, index=False)
    pd.DataFrame(index_test, columns=['index']).to_csv(test_path, index=False)

    print(f"\nIndexes saved at: {output_dir}")
    print(f"Train: {train_path}")
    print(f"Validation: {val_path}")
    print(f"Test: {test_path}")
    print(f"Train samples: {len(index_train)}, Validation samples: {len(index_val)}, Test samples: {len(index_test)}")


# ===== Training and Evaluation Functions =====

# Simple training with train and validation sets

def train(model, dataloader, optimizer, loss_fn, device):
    """
    A simple train function. Performs a single training epoch for a graph neural network.

    Args:
        model (torch.nn.Module): An instantiated model to train.
        dataloader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the training set.
        optimizer (torch.optim.Optimizer): A Torch optimizer.
        loss_fn: A Torch loss function. e.g., nn.MSELoss.
        device (torch.device): cuda or cpu.
    
    Returns:
        float: The total average loss over the entire dataset.
    """

    model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs # loss.item() is the average loss for the batch * number of graphs on batch
        optimizer.step()
        
    return total_loss / len(dataloader.dataset) # the total average loss


@torch.no_grad()
def test(model, dataloader, loss_fn, device):
    """
    A simple test function. Measures the loss function on a dataset.

    Args:
        model (torch.nn.Module): An instantiated model to test.
        dataloader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the dataset.
        loss_fn: A Torch loss function. e.g., nn.MSELoss.
        device (torch.device): cuda or cpu.
    
    Returns:
        float: The total average loss over the entire dataset.
    """

    model.eval()
    total_loss = 0

    for data in dataloader:
        data = data.to(device)
        outputs = model(data)
        loss = loss_fn(outputs, data.y)
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset) # the total average error


def train_with_early_stopping(model, model_name, output_dir, filename_prefix, train_loader, val_loader, epochs, lr, device, patience = 10, filename_ending=''):
    """
    Runs the training process with early stopping using training and validation sets.
    The model is trained on the training dataset and evaluated on the validation dataset.
    Saves the best model to the output directory. Continually overwrites the previous
    model when a new better one is trained. Increases the patience counter when a new
    model is not better than the previous one.

    Stops when max epoch number or max patience number is achieved.

    Uses Mean Squared Error loss function.

    Output schema: {filename_prefix}_{model_name}{filename_ending}.pt

    Args:
        model (torch.nn.Module): An instantiated model to train.
        model_name (str): Model name to save.
        output_dir (str): Path to the directory to save the models.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        train_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the training dataset.
        val_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the validation dataset.
        epochs (int): Number of maximum epochs to run training.
        lr (float): Learning rate hyperparameter.
        device (torch.device): cuda or cpu.
        patience (int): Number of epochs to wait for improvement before stopping.
        filename_ending (str, optional): The output files suffix.
    
    Returns:
        Tuple[nn.Module, Dict[str, List[float]]] model, history:
        model: The trained model with updated weights.
        history (dict): train_loss and val_loss history lists for plotting.
    """

    if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

    print(f"\n----- Training {model_name} with Early Stopping (Patience={patience}) -----")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # ------------- loss function -------------

    os.makedirs(f'{output_dir}', exist_ok=True)
    best_model_path = output_dir / f'{filename_prefix}_{model_name}{filename_ending}.pt'

    best_val_loss = None
    patience_counter = 0

    # Keep history for plotting
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss  = test(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
             print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Error: {val_loss:.6f}")

        if best_val_loss is None or val_loss <= best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reset counter on improvement
            # Save the best model
            os.makedirs(f'{output_dir}', exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1 # Increase counter if no improvement

        if patience_counter >= patience:
            print(f"--- Early stopping triggered at epoch {epoch} ---")
            break

    print(f"----- Finished training {model_name}. Best Val Error (MSE): {best_val_loss:.6f} -----")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    return model, history


def objective(trial,
              model_class,
              train_loader,
              val_loader,
              node_features,
              edge_features,
              epochs,
              hidden_dim_list,
              dropout_range,
              lr_range,
              device,
              patience = 10):
    """
    The Optuna objective function for a single hyperparameter tuning trial.
    Same as train_with_early_stopping but for the Optuna Hyperparameter Tuning.
    It suggests new values to the hyperparameters for each trial. Each trial is 
    scored by its best validation loss. 

    Args:
        model_class (torch.nn.Module): A model class to use for tuning.
        train_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the training dataset.
        val_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the validation dataset.
        node_features (int): Node features size
        edge_features (int): Edge features size
        epochs (int): Number of maximum epochs to run training.
        hidden_dim_list (list[int]): List of discrete values to test for the hidden dimension hyperparameter.
        dropout_range ([float, float]): List of the minimum and maximum value to test for the dropout range hyperparameter.
        lr_range ([float, float]): List of the minimum and maximum value to test for the learning rate hyperparameter.
        device (torch.device): cuda or cpu.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        float: The validation loss value of the best model.
    """
    # Suggest Hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
    dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
    lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
    
    # Instantiate the model
    model = model_class(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        output_dim=1,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # ------------- loss function -------------

    best_val_loss = float('inf')
    patience_counter = 0

    # Early stopping mechanism is still used for efficiency
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, loss_fn, device) # just training - not saving the model
        val_loss = test(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Trial {trial.number} early stopped at epoch {epoch}")
            break
            
        # Pruning tells Optuna to stop bad trials early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def objective_with_globals(trial,
              model_class,
              train_loader,
              val_loader,
              node_features,
              edge_features,
              global_features,
              epochs,
              hidden_dim_list,
              dropout_range,
              lr_range,
              device,
              patience = 10):
    """
    The Optuna objective function for a single hyperparameter tuning trial.
    Same as train_with_early_stopping but for the Optuna Hyperparameter Tuning.
    It suggests new values to the hyperparameters for each trial. Each trial is 
    scored by its best validation loss. Adds support for models with global
    features.

    Args:
        model_class (torch.nn.Module): A model class to use for tuning.
        train_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the training dataset.
        val_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the validation dataset.
        node_features (int): Node features size
        edge_features (int): Edge features size
        global_features (int): Global features size
        epochs (int): Number of maximum epochs to run training.
        hidden_dim_list (list[int]): List of discrete values to test for the hidden dimension hyperparameter.
        dropout_range ([float, float]): List of the minimum and maximum value to test for the dropout range hyperparameter.
        lr_range ([float, float]): List of the minimum and maximum value to test for the learning rate hyperparameter.
        device (torch.device): cuda or cpu.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        float: The validation loss value of the best model.
    """

    # Suggest Hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
    dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
    lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
    
    # Instantiate the model
    model = model_class(
        node_features=node_features,
        edge_features=edge_features,
        global_features=global_features,
        hidden_dim=hidden_dim,
        output_dim=1,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # ------------- loss function -------------

    best_val_loss = float('inf')
    patience_counter = 0

    # Early stopping mechanism is still used for efficiency
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, loss_fn, device) # just training - not saving the model
        val_loss = test(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Trial {trial.number} early stopped at epoch {epoch}")
            break
            
        # Pruning tells Optuna to stop bad trials early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


# ----------- ElasticNet -------------

# ElasticNet src:
#     Minimizes the objective function::

#             1 / (2 * n_samples) * ||y - Xw||^2_2
#             + alpha * l1_ratio * ||w||_1
#             + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    # (This is the same as 0.5 * MSE + alpha * l1_ratio * abs(w) + 0.5 * alpha * (1-l1_ratio) * (w^2))

    # If you are interested in controlling the L1 and L2 penalty
    # separately, keep in mind that this is equivalent to::

    #         a * ||w||_1 + 0.5 * b * ||w||_2^2

    # where::

    #         alpha = a + b and l1_ratio = a / (a + b)


def train_ElasticNet(model, dataloader, optimizer, device, alpha, l1_ratio=1.0):
    """
    A simple train function with an ElasticNet loss function.
    Performs a single training epoch for a graph neural network.
    Default value for L1 ratio is 1.

    Args:
        model (torch.nn.Module): An instantiated model to train.
        dataloader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the training set.
        optimizer (torch.optim.Optimizer): A Torch optimizer.
        device (torch.device): cuda or cpu.
        alpha (float): ElasticNet alpha hyperparameter
        l1_ratio (float): ElasticNet L1 ratio hyperparameter
    
    Returns:
        float: The total average loss over the entire dataset.
    """

    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss_fn = nn.MSELoss() 
        mse_loss = loss_fn(outputs, data.y)

        l1_penalty = 0
        l2_penalty = 0
        for param in model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
            l2_penalty += torch.sum(param.pow(2.0))

        loss = 0.5 * mse_loss + alpha * l1_ratio * l1_penalty + 0.5 * alpha * (1 - l1_ratio) * l2_penalty

        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def test_ElasticNet(model, dataloader, device):
    """
    A simple test function. Measures the loss function on a dataset.
    Uses Mean Squared Error loss function.

    Args:
        model (torch.nn.Module): An instantiated model to test.
        dataloader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the dataset.
        device (torch.device): cuda or cpu
    
    Returns:
        float: The total average loss over the entire dataset.
    """
    model.eval()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        outputs = model(data)
        loss_fn = nn.MSELoss() 
        mse_loss = loss_fn(outputs, data.y)
        total_loss += mse_loss.item() * data.num_graphs
    return total_loss / len(dataloader.dataset) # the total average error



def objective_ElasticNet(trial,
              model_class,
              train_loader,
              val_loader,
              node_features,
              edge_features,
              epochs,
              hidden_dim_list,
              lr_range,
              alpha_range,
              device,
              l1_ratio=1.0,
              patience = 10):
    """
    The Optuna objective function for a single ElasticNet regularization
    hyperparameter tuning trial. Same as train_with_early_stopping but
    for the Optuna Hyperparameter Tuning. It suggests new values to the
    hyperparameters for each trial. Each trial is scored by its best
    validation loss. 

    Args:
        model_class (torch.nn.Module): A model class to use for tuning.
        train_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the training dataset.
        val_loader (torch_geometric.loader.DataLoader): A Torch geometric dataloader for the validation dataset.
        node_features (int): Node features size
        edge_features (int): Edge features size
        epochs (int): Number of maximum epochs to run training.
        hidden_dim_list (list[int]): List of discrete values to test for the hidden dimension hyperparameter.
        dropout_range ([float, float]): List of the minimum and maximum value to test for the dropout range hyperparameter.
        lr_range ([float, float]): List of the minimum and maximum value to test for the learning rate hyperparameter.
        device (torch.device): cuda or cpu.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        float: The validation loss value of the best model.
    """
    # Suggest Hyperparameters
    # Dropped the dropout rate tuning because it is conflincting with regularization of the alpha variable
    hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
    lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
    alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], log=True)
    
    # Instantiate the model
    model = model_class(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        output_dim=1,
        dropout_rate=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    # Early stopping mechanism is still used for efficiency
    for epoch in range(1, epochs + 1):
        train_ElasticNet(model, train_loader, optimizer, device, alpha, l1_ratio) # just training - not saving the model
        val_loss = test_ElasticNet(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Trial {trial.number} early stopped at epoch {epoch}")
            break
            
        # Pruning tells Optuna to stop bad trials early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


# ===== Evaluate and History plot functions =====

# ----------- These were made with A.I. -----------

def evaluate_and_plot(model, model_name, loader, device, scalers, prop_names):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())
            all_true.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # --- Inverse transform to get results in original units ---
    unscaled_preds = np.zeros_like(all_preds)
    unscaled_true = np.zeros_like(all_true)

    print(f"\n----- Results for {model_name} -----")
    for i, name in enumerate(prop_names):
        scaler = scalers[name]
        # Reshape for scaler which expects 2D array
        unscaled_preds[:, i] = scaler.inverse_transform(all_preds[:, i].reshape(-1, 1)).flatten()
        unscaled_true[:, i] = scaler.inverse_transform(all_true[:, i].reshape(-1, 1)).flatten()

        mse = mean_squared_error(unscaled_true[:, i], unscaled_preds[:, i])
        print(f"  - {name} MSE: {mse:.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, len(prop_names), figsize=(18, 5))
    fig.suptitle(f'{model_name}: True vs. Predicted Values', fontsize=16)

    for i, (ax, name) in enumerate(zip(axes, prop_names)):
        ax.scatter(unscaled_true[:, i], unscaled_preds[:, i], alpha=0.3, s=10)

        # Add a y=x line for reference
        limits = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(limits, limits, color='red', linestyle='--', label='Perfect Prediction')

        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_training_history(history, model_name):
    """
    Plots the training and validation loss curves from a history dictionary.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and validation loss
    ax.plot(epochs, history['train_loss'], 'o-', label='Training Loss', alpha=0.8)
    ax.plot(epochs, history['val_loss'], 'o-', label='Validation Loss', alpha=0.8)

    # Find the epoch with the best validation loss for annotation
    best_val_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])

    # Add a vertical line and annotation for the best epoch
    ax.axvline(best_val_epoch, color='red', linestyle='--', lw=1, label=f'Best Epoch ({best_val_epoch})')
    ax.annotate(f'Best Val Loss: {best_val_loss:.4f}',
                xy=(best_val_epoch, best_val_loss),
                xytext=(best_val_epoch + 5, best_val_loss + 0.05 * best_val_loss), # Offset for readability
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))

    # Formatting
    ax.set_title(f'{model_name}: Training and Validation Loss', fontsize=16)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Ensure y-axis starts from a reasonable place if losses are small
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()



def calculate_mae(model, model_name, loader, device, scalers, prop_names):
    """
    Loads a model, evaluates it on a data loader, and calculates the
    Mean Absolute Error (MAE) for each property in original units.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())
            all_true.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # Inverse transform to get results in original units
    unscaled_preds = np.zeros_like(all_preds)
    unscaled_true = np.zeros_like(all_true)

    print(f"\n----- Mean Absolute Error (MAE) for {model_name} -----")
    for i, name in enumerate(prop_names):
        scaler = scalers[name]
        # Reshape for scaler which expects 2D array
        unscaled_preds[:, i] = scaler.inverse_transform(all_preds[:, i].reshape(-1, 1)).flatten()
        unscaled_true[:, i] = scaler.inverse_transform(all_true[:, i].reshape(-1, 1)).flatten()

        mae = mean_absolute_error(unscaled_true[:, i], unscaled_preds[:, i])
        print(f"  - {name} MAE: {mae:.4f}")

    return


def evaluate_and_plot_one_property(model, model_name, loader, device, scalers, prop_name):
    """
    (Single-Property Version)
    Evaluates a model and plots the results for a single target property.
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())
            all_true.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # --- Inverse transform to get results in original units ---
    scaler = scalers[prop_name]
    unscaled_preds = scaler.inverse_transform(all_preds)
    unscaled_true = scaler.inverse_transform(all_true)

    print(f"\n----- Results for {model_name} -----")
    mse = mean_squared_error(unscaled_true, unscaled_preds)
    print(f"  - {prop_name} MSE: {mse}")

    # --- Plotting ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5)) # Changed to single subplot
    fig.suptitle(f'{model_name}: True vs. Predicted Values', fontsize=16)

    ax.scatter(unscaled_true, unscaled_preds, alpha=0.3, s=10)

    # Add a y=x line for reference
    limits = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(limits, limits, color='red', linestyle='--', label='Perfect Prediction')

    ax.set_xlabel(f"True {prop_name}")
    ax.set_ylabel(f"Predicted {prop_name}")
    ax.set_title(prop_name)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_training_history_one_property(history, model_name, prop_name):
    """
    (Single-Property Version)
    Plots the training and validation loss curves from a history dictionary.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'o-', label='Training Loss', alpha=0.8)
    ax.plot(epochs, history['val_loss'], 'o-', label='Validation Loss', alpha=0.8)

    best_val_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])

    ax.axvline(best_val_epoch, color='red', linestyle='--', lw=1, label=f'Best Epoch ({best_val_epoch})')
    ax.annotate(f'Best Val Loss: {best_val_loss:.4f}',
                xy=(best_val_epoch, best_val_loss),
                xytext=(best_val_epoch + 5, best_val_loss + 0.05 * best_val_loss),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))

    ax.set_title(f'{model_name} ({prop_name}): Training and Validation Loss', fontsize=16)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def calculate_mae_one_property(model, model_name, loader, device, scalers, prop_name):
    """
    (Single-Property Version)
    Calculates and prints the Mean Absolute Error (MAE) for a single property.
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())
            all_true.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # Inverse transform
    scaler = scalers[prop_name]
    unscaled_preds = scaler.inverse_transform(all_preds)
    unscaled_true = scaler.inverse_transform(all_true)

    print(f"\n----- Mean Absolute Error (MAE) for {model_name} -----")
    mae = mean_absolute_error(unscaled_true, unscaled_preds)
    print(f"  - {prop_name} MAE: {mae}")



# ===== Global features feature selection =====

# def calculate_global_feature_importances(names_filepath,
#                                          filename_prefix,
#                                          prop_names,
#                                          dataset_naming_schema,
#                                          dataset_dir,
#                                          max_depth_range,
#                                          lr_range,
#                                          n_estimators_range,
#                                          outer_folds,
#                                          inner_folds,
#                                          output_dir,
#                                          filename_ending=''
#                                          ):
#     """
#     Performs importance analysis for global features using an XGBoost model with
#     nested cross-validation. Uses GridSearchCV to find the optimal hyperparameters.

#     Outputs a JSON file containing a dictionary where each key is a property name,
#     and the value is another dictionary mapping each global feature name to its
#     averaged importance score.

#     Output schema: {filename_prefix}_global_features_importances{filename_ending}.json

#     Args:
#         names_filepath (str): Path to the JSON file containing the list of feature names.
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
#         dataset_naming_schema (str): The name for the global dataset files. e.g., 'qm9_global_dataset'.
#         dataset_dir (str): Path to the directory containing the global dataset files.
#         max_depth_list (List[int]): List of values to test for the `max_depth` hyperparameter.
#         learning_rate_list (List[float]): List values to test for the `learning_rate` hyperparameter.
#         n_estimators_list (List[int]): List values to test for the `n_estimators` hyperparameter.
#         outer_folds (int): The number of splits for the outer cross-validation loop.
#         inner_folds (int): The number of splits for the inner cross-validation loop.
#         output_dir (str): Path to the directory to save the JSON file.
#         filename_ending (str, optional): The output files suffix.
#     """

#     try:
#         with open(names_filepath, 'r') as f:
#             feature_names_data = json.load(f)
#         global_feature_names = feature_names_data['global_features_names']
#         print(f"Loaded {len(global_feature_names)} total global feature names.")
#     except FileNotFoundError:
#         print(f"ERROR: Feature name file not found at {names_filepath}")
#         print(f"Run save_feature_names from the data preprocessing notebook")


#     global_feature_selection_importances = {}

#     for prop_name in prop_names:
#         print(f"\nCalculating Global Feature Importance for: {prop_name}...")

#         # Data Loading
#         dataset_filename = f'{dataset_naming_schema}_{prop_name}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             full_dataset = torch.load(dataset_path, weights_only=False)
#         except FileNotFoundError:
#             print(f"ERROR: Dataset not found at {dataset_path}")
#             print(f"Run convert_smiles_to_graph_with_globals from the preprocessing notebook")


#         # Create arrays from dataset (x = global features, y = target values)
#         x = np.array([data.u.numpy().flatten() for data in tqdm(full_dataset)])
#         y = np.array([data.y.numpy().flatten() for data in tqdm(full_dataset)]).ravel()

#         # Convert NANs and INFs to 0
#         x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

#         # Parameter grid for XGBoost grid search
#         param_grid = {
#         'max_depth': max_depth_range,
#         'learning_rate': lr_range,
#         'n_estimators': n_estimators_range,
#         'subsample': [0.8],
#         'colsample_bytree': [0.8],
#         }

#         # Implement nested cross validation
#         outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
#         inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

#         outer_importances = []

#         # ------- Train the XGBoost model -------
#         print(" Training XGBoost model...")

#         for i, (train_index, test_index) in enumerate(outer_cv.split(x)):
#             x_train, x_test = x[train_index], x[test_index]
#             y_train, y_test = y[train_index], y[test_index]

#             print(f'Fold {i+1} out of {outer_folds}')

#             if torch.cuda.is_available():
#                 device = "cuda"
#                 tree_method = "approx"

#                 xgb_model = xgb.XGBRegressor(
#                     objective='reg:squarederror',
#                     random_state=42,
#                     device=device,
#                     tree_method=tree_method,
#                 )

#             else:
#                 device = "cpu"
#                 tree_method = "auto"

#                 xgb_model = xgb.XGBRegressor(
#                     objective='reg:squarederror',
#                     random_state=42,
#                     device=device,
#                     tree_method=tree_method,
#                 )

#             # IMPORTANT
#             # Use only one core for the grid search otherwise it will crash.
#             # That way only one fold is loaded into memory.
#             # It is actually faster using only one core (xbg is running on gpu, so it doesn't slow down)
#             grid_search = GridSearchCV(
#                 estimator=xgb_model,
#                 param_grid=param_grid,
#                 scoring='neg_mean_squared_error',
#                 n_jobs=1,
#                 cv=inner_cv,
#                 verbose=0,
#             )

#             grid_search.fit(x_train,y_train)

#             best_model = grid_search.best_estimator_
#             best_model.fit(x_train, y_train)

#             outer_importances.append(best_model.feature_importances_)

#         # Average the importance to generalize feature importance (not just the max)
#         # Threshold is half of the max importance
#         avg_importances = np.mean(outer_importances, axis=0)

#         global_importance_dict = {name: float(importance) for name, importance in zip(global_feature_names, avg_importances)}
#         global_feature_selection_importances[prop_name] = {'global_features_importances': global_importance_dict}


#     os.makedirs(output_dir, exist_ok=True) 
#     importance_filename = f'{filename_prefix}_global_features_importances{filename_ending}.json'
#     importance_filepath = output_dir / importance_filename
#     with open(importance_filepath, 'w') as f:
#         json.dump(global_feature_selection_importances, f, indent=4)

#     print("\n----- Computed global feature importances -----")    
#     print(f"Feature importances saved to '{importance_filepath}'")
    

def calculate_global_feature_importances(names_filepath,
                                         filename_prefix,
                                         prop_name,
                                         dataset_naming_schema,
                                         dataset_dir,
                                         max_depth_range,
                                         lr_range,
                                         n_estimators_range,
                                         outer_folds,
                                         inner_folds,
                                         output_dir,
                                         filename_ending=''
                                         ):
    """
    Performs importance analysis for global features using an XGBoost model with
    nested cross-validation. Uses GridSearchCV to find the optimal hyperparameters.

    Outputs a JSON file containing a dictionary where each key is a property name,
    and the value is another dictionary mapping each global feature name to its
    averaged importance score.

    Output schema: {filename_prefix}_global_features_importances_{prop_name}{filename_ending}.json

    Args:
        names_filepath (str): Path to the JSON file containing the list of feature names.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_name (str): The target property name to analyze. e.g., 'Dipole_moment'
        dataset_naming_schema (str): The name for the global dataset files. e.g., 'qm9_global_dataset'.
        dataset_dir (str): Path to the directory containing the global dataset files.
        max_depth_list (List[int]): List of values to test for the `max_depth` hyperparameter.
        learning_rate_list (List[float]): List values to test for the `learning_rate` hyperparameter.
        n_estimators_list (List[int]): List values to test for the `n_estimators` hyperparameter.
        outer_folds (int): The number of splits for the outer cross-validation loop.
        inner_folds (int): The number of splits for the inner cross-validation loop.
        output_dir (str): Path to the directory to save the JSON file.
        filename_ending (str, optional): The output files suffix.
    """

    try:
        with open(names_filepath, 'r') as f:
            feature_names_data = json.load(f)
        global_feature_names = feature_names_data['global_features_names']
        print(f"Loaded {len(global_feature_names)} total global feature names.")
    except FileNotFoundError:
        print(f"ERROR: Feature name file not found at {names_filepath}")
        print(f"Run save_feature_names from the data preprocessing notebook")


    print(f"\nCalculating Global Feature Importance for: {prop_name}...")

    # Data Loading
    dataset_filename = f'{dataset_naming_schema}_{prop_name}.pt'
    dataset_path = dataset_dir / dataset_filename
    try:
        full_dataset = torch.load(dataset_path, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Run convert_smiles_to_graph_with_globals from the preprocessing notebook")


    # Create arrays from dataset (x = global features, y = target values)
    x = np.array([data.u.numpy().flatten() for data in tqdm(full_dataset)])
    y = np.array([data.y.numpy().flatten() for data in tqdm(full_dataset)]).ravel()

    # Convert NANs and INFs to 0
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Parameter grid for XGBoost grid search
    param_grid = {
    'max_depth': max_depth_range,
    'learning_rate': lr_range,
    'n_estimators': n_estimators_range,
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    }

    # Implement nested cross validation
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    outer_importances = []

    # ------- Train the XGBoost model -------
    print(" Training XGBoost model...")

    for i, (train_index, test_index) in enumerate(outer_cv.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f'Fold {i+1} out of {outer_folds}')

        if torch.cuda.is_available():
            device = "cuda"
            tree_method = "approx"

            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                device=device,
                tree_method=tree_method,
            )

        else:
            device = "cpu"
            tree_method = "auto"

            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                device=device,
                tree_method=tree_method,
            )

        # IMPORTANT
        # Use only one core for the grid search otherwise it will crash.
        # That way only one fold is loaded into memory.
        # It is actually faster using only one core (xbg is running on gpu, so it doesn't slow down)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            cv=inner_cv,
            verbose=0,
        )

        grid_search.fit(x_train,y_train)

        best_model = grid_search.best_estimator_
        best_model.fit(x_train, y_train)

        outer_importances.append(best_model.feature_importances_)

    # Average the importance to generalize feature importance (not just the max)
    # Threshold is half of the max importance
    avg_importances = np.mean(outer_importances, axis=0)

    global_importance_dict = {name: float(importance) for name, importance in zip(global_feature_names, avg_importances)}
    importances = {prop_name: {'global_features_importances': global_importance_dict}}


    os.makedirs(output_dir, exist_ok=True) 
    importance_filename = f'{filename_prefix}_global_features_importances_{prop_name}{filename_ending}.json'
    importance_filepath = output_dir / importance_filename
    with open(importance_filepath, 'w') as f:
        json.dump(importances, f, indent=4)

    print(f"\n----- Computed global feature importances for {prop_name}-----")    
    print(f"Feature importances saved to '{importance_filepath}'")


# def select_global_features(names_filepath,
#                            filename_prefix,
#                            threshold_fraction,
#                            output_dir,
#                            filename_ending=''
#                            ):
#     """
#     Selects the most important global features based on the
#     calculate_global_feature_importances function.

#     For each property, it identifies the maximum importance score and applies a
#     threshold based on the provided `threshold_fraction` * max_importance.
#     Features with an importance score above this threshold are considered "selected".

#     threshold_fraction should be between 0 and 1. 

#     The output is a JSON file containing the names and original indices
#     of only the selected features for each property, ready for use in dataset
#     pruning.

#     Output schema: {filename_prefix}_global_features_selection_results{filename_ending}.json

#     Args:
#         names_filepath (str): Path to the JSON file containing the list of feature names.
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         threshold_fraction (float): The fraction of the maximum importance score
#             to use as the selection threshold. e.g., 0.5 means features with at
#             least 50% of the max importance will be kept.
#         output_dir (str): Path to the directory to save the JSON file.
#         filename_ending (str, optional): The output files suffix.
#     """

#     try:
#         with open(names_filepath, 'r') as f:
#             feature_names_data = json.load(f)
#         global_feature_names = feature_names_data['global_features_names']
#         print(f"Loaded {len(global_feature_names)} total global feature names.")
#     except FileNotFoundError:
#         print(f"ERROR: Feature name file not found at {names_filepath}")
#         print(f"Run save_feature_names from the data preprocessing notebook")

#     importance_filename = f'{filename_prefix}_global_features_importances{filename_ending}.json'
#     importance_filepath = output_dir / importance_filename
#     try:
#         with open(importance_filepath, 'r') as f:
#             global_feature_selection_importances = json.load(f)
#         print(f"Loaded importances from '{importance_filepath}'")
#     except FileNotFoundError:
#         print(f"ERROR: Feature importance file not found at {importance_filepath}")
#         print(f"Run calculate_global_feature_importances models analysis notebook")

#     global_selection_results = {}

#     for prop_name, data_dict in global_feature_selection_importances.items():
#         print(f"\n======= Global feature selection for: {prop_name} =======")

#         importance_dict = data_dict['global_features_importances']
#         importance_values = np.array(list(importance_dict.values()))
#         max_importance = np.max(importance_values)
#         # threshold_fraction = 0.3 # ------------------ make this a func var ---------------------
#         threshold = threshold_fraction * max_importance
    
#         selected_indices = [i for i, importance in enumerate(importance_values) if importance >= threshold]
#         selected_names = [global_feature_names[i] for i in selected_indices]

#         global_selection_results[prop_name] = {
#             'global_features_index': selected_indices,
#             'global_features_selected_names': selected_names,
#         }

#         print(f"\nGlobal feature importance for {prop_name}")
#         print(f"Max importance score: {max_importance:.5f}")
#         print(f"Threshold fraction: {threshold_fraction}")
#         print(f"Selection threshold: {threshold:.5f}")
#         print(f"Total global features selected: {len(selected_indices)} out of {len(global_feature_names)}")
#         print("Top 10 global features:")
#         sorted_indices = np.argsort(importance_values)[::-1]
#         for i in sorted_indices[:10]:
#             print(f"       - {global_feature_names[i]}: {importance_values[i]:.4f}")

#     # Save the results
#     os.makedirs(output_dir, exist_ok=True) 
#     selection_filename = f'{filename_prefix}_global_features_selection_results{filename_ending}.json'
#     selection_filepath = output_dir / selection_filename
#     with open(selection_filepath, 'w') as f:
#         json.dump(global_selection_results, f, indent=4)
#     print(f"\nGlobal features selection results saved to '{selection_filepath}'")


def select_global_features(names_filepath,
                           filename_prefix,
                           prop_name,
                           threshold_fraction,
                           output_dir,
                           filename_ending=''
                           ):
    """
    Selects the most important global features based on the
    calculate_global_feature_importances function.

    For each property, it identifies the maximum importance score and applies a
    threshold based on the provided `threshold_fraction` * max_importance.
    Features with an importance score above this threshold are considered "selected".

    threshold_fraction should be between 0 and 1. 

    The output is a JSON file containing the names and original indices
    of only the selected features for each property, ready for use in dataset
    pruning.

    Output schema: {filename_prefix}_global_features_selection_{prop_name}{filename_ending}.json

    Args:
        names_filepath (str): Path to the JSON file containing the list of feature names.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_name (str): The target property name to analyze. e.g., 'Dipole_moment'
        threshold_fraction (float): The fraction of the maximum importance score
            to use as the selection threshold. e.g., 0.5 means features with at
            least 50% of the max importance will be kept.
        output_dir (str): Path to the directory to save the JSON file.
        filename_ending (str, optional): The output files suffix.
    """

    try:
        with open(names_filepath, 'r') as f:
            feature_names_data = json.load(f)
        global_feature_names = feature_names_data['global_features_names']
        print(f"Loaded {len(global_feature_names)} total global feature names.")
    except FileNotFoundError:
        print(f"ERROR: Feature name file not found at {names_filepath}")
        print(f"Run save_feature_names from the data preprocessing notebook")

    importance_filename = f'{filename_prefix}_global_features_importances_{prop_name}{filename_ending}.json'
    importance_filepath = output_dir / importance_filename
    try:
        with open(importance_filepath, 'r') as f:
            global_feature_selection_importances = json.load(f)
        print(f"Loaded importances from '{importance_filepath}'")
    except FileNotFoundError:
        print(f"ERROR: Feature importance file not found at {importance_filepath}")
        print(f"Run calculate_global_feature_importances models analysis notebook")

    global_selection_results = {}

    print(f"\n======= Global feature selection for: {prop_name} =======")

    data_dict = global_feature_selection_importances[prop_name]
    importance_dict = data_dict['global_features_importances']
    importance_values = np.array(list(importance_dict.values()))
    max_importance = np.max(importance_values)
    # threshold_fraction = 0.3 # ------------------ make this a func var ---------------------
    threshold = threshold_fraction * max_importance
    
    selected_indices = [i for i, importance in enumerate(importance_values) if importance >= threshold]
    selected_names = [global_feature_names[i] for i in selected_indices]

    global_selection_results = {
        prop_name: {'global_features_index': selected_indices,
                    'global_features_selected_names': selected_names,
                    }
    }

    print(f"\nGlobal feature importance for {prop_name}")
    print(f"Max importance score: {max_importance:.5f}")
    print(f"Threshold fraction: {threshold_fraction}")
    print(f"Selection threshold: {threshold:.5f}")
    print(f"Total global features selected: {len(selected_indices)} out of {len(global_feature_names)}")
    print("Top 10 global features:")
    sorted_indices = np.argsort(importance_values)[::-1]
    for i in sorted_indices[:10]:
        print(f"       - {global_feature_names[i]}: {importance_values[i]:.4f}")

    # Save the results
    os.makedirs(output_dir, exist_ok=True) 
    selection_filename = f'{filename_prefix}_global_features_selection_{prop_name}{filename_ending}.json'
    selection_filepath = output_dir / selection_filename
    with open(selection_filepath, 'w') as f:
        json.dump(global_selection_results, f, indent=4)
    print(f"\nGlobal features selection results saved to '{selection_filepath}'")


# ===== Node and Edge Features - Feature selection - ElasticNet (Lasso, l1_ratio=1) =====


# def elasticnet_feature_selection(names_filepath,
#                                  filename_prefix,
#                                  prop_names,
#                                  dataset_naming_schema,
#                                  dataset_dir,
#                                  model_class,
#                                  epochs,
#                                  hidden_dim_list,
#                                  dropout_range,
#                                  lr_range,
#                                  alpha_range,
#                                  device,
#                                  n_trials,
#                                  outer_folds,
#                                  inner_folds,
#                                  batch_size,
#                                  output_dir,
#                                  filename_ending='',
#                                  l1_ratio=1.0,
#                                  ):
#     """
#     Performs importance analysis for node & edge features using a given model with
#     nested cross-validation. Uses the objective_ElasticNet, an Optuna objective
#     function for a single ElasticNet regularization hyperparameter tuning trial,
#     to find the optimal hyperparameters.

#     The ElasticNet regularization is run for n_trials inside the inner cv fold.

#     Outputs a JSON file containing a dictionary where each key is a property name,
#     and the value is another dictionary mapping each node & edge feature name to
#     its averaged importance score.

#     Output schema: {filename_prefix}_node_edge_features_importances{filename_ending}.json

#     Args:
#         names_filepath (str): Path to the JSON file containing the list of feature names.
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
#         dataset_naming_schema (str): The name for the dataset files. e.g., 'qm9_global_dataset'.
#         dataset_dir (str): Path to the directory containing the dataset files.
#         model_class (torch.nn.Module): A model class to use for tuning.
#         epochs (int): Number of maximum epochs to run ElasticNet training without early stopping.
#         patience (int): Number of epochs to wait for improvement before stopping ElasticNet training.
#         hidden_dim_list  (List[int]): List of hidden dimensions to try. e.g., [128, 256].
#         dropout_range (List[float]): List of the minimum and maximum value to test for the `dropout_rate` hyperparameter.
#         lr_range (List[float]): List of the minimum and maximum value to test for the `learning_rate` hyperparameter.
#         alpha_range (List[float]): List of the minimum and maximum value to test for the `alpha` ElastiNet hyperparameter.
#         device (torch.device): cuda or cpu.
#         n_trials (int): The number of Optuna trials to run for each inner loop.
#         outer_folds (int): The number of splits for the outer cross-validation loop.
#         inner_folds (int): The number of splits for the inner cross-validation loop.
#         output_dir (str): Path to the directory to save the JSON file.
#         filename_ending (str, optional): The output files suffix.
#         l1_ratio (float, optional): The L1 ratio for Elastic Net. Defaults to 1.0 (Lasso).
#     """

#     try:
#         with open(names_filepath, 'r') as f:
#             feature_names_data = json.load(f)
#         node_feature_names = feature_names_data['node_features_names']
#         edge_feature_names = feature_names_data['edge_features_names']
#         print(f"Successfully loaded feature names from {names_filepath}")
#     except FileNotFoundError:
#         print(f"ERROR: Feature name file not found at {names_filepath}")
#         print("Run save_feature_names from the data preprocessing notebook.")

#     feature_selection_importances = {}

#     for prop_name in prop_names:
#         print(f"\n======= Feature Selection for: {prop_name} =======")

#         # Data Loading
#         dataset_filename = f'{dataset_naming_schema}_{prop_name}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             full_dataset = torch.load(dataset_path, weights_only=False)
#         except FileNotFoundError:
#             print(f"ERROR: Dataset not found at {dataset_path}")
#             print(f"Run convert_smiles_to_graph from the  preprocessing notebook")

#         # Determine feature sizes from the first graph
#         num_node_features = full_dataset[0].num_node_features
#         num_edge_features = full_dataset[0].num_edge_features

#         # Implement nested cross validation
#         outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
#         inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

#         outer_node_importances = []
#         outer_edge_importances = []

#         # ------- Start nested cross-validation -------
#         print(" Starting nested cross-validation...")

#         for fold_idx, (outer_train_index, outer_test_index) in enumerate(outer_cv.split(full_dataset)):
#             print(f'Fold {fold_idx+1} out of {outer_folds}')
#             outer_train_dataset = Subset(full_dataset, outer_train_index)


#             # Optuna Hyperparameter Study on the outer train_val set
#             print(" Running Optuna to find best hyperparameters for this fold...")

#             # We define the objective function inside the loop to capture the correct dataset split
#             def objective_inner_cv(trial):
#                 # Suggest Hyperparameters
#                 hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
#                 dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
#                 lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
#                 alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], log=True)

#                 val_losses = []

#                 # Inner loop
#                 for inner_train_idx, inner_val_idx in inner_cv.split(outer_train_dataset):

#                     inner_train_dataset = Subset(outer_train_dataset, inner_train_idx)
#                     inner_val_dataset = Subset(outer_train_dataset, inner_val_idx)

#                     inner_train_loader = DataLoader(inner_train_dataset, batch_size=batch_size, shuffle=True)
#                     inner_val_loader = DataLoader(inner_val_dataset, batch_size=batch_size, shuffle=False)

#                     model = model_class(
#                         node_features=num_node_features,
#                         edge_features=num_edge_features,
#                         hidden_dim=hidden_dim,
#                         output_dim=1,
#                         dropout_rate=dropout_rate
#                     ).to(device)
#                     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#                     # Train for a smaller, fixed number of epochs for speed
#                     for _ in range(60):
#                         train_ElasticNet(model, inner_train_loader, optimizer, device, alpha, l1_ratio)

#                     val_loss = test_ElasticNet(model, inner_val_loader, device)
#                     val_losses.append(val_loss)

#                 return np.mean(val_losses)

#             study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
#             study.optimize(objective_inner_cv, n_trials=n_trials, show_progress_bar=True)

#             best_params = study.best_params
#             print(f"Best hyperparameters for {prop_name}: {best_params}")


#             # Retrain the model with the best alpha to get the final weights for inspection
#             print("\nRetraining with best alpha to get final weights...")

#             trained_ElasticNet_model = model_class(
#                 node_features=num_node_features,
#                 edge_features=num_edge_features,
#                 hidden_dim=best_params['hidden_dim'],
#                 output_dim=1,
#                 dropout_rate=best_params['dropout_rate']
#             ).to(device)
#             optimizer = torch.optim.Adam(trained_ElasticNet_model.parameters(), lr=best_params['lr'])

#             outer_train_loader = DataLoader(outer_train_dataset, batch_size=batch_size, shuffle=True)
    
#             # Train for longer in order to push unimportant params to zero
#             for _ in range(epochs):
#                 train_ElasticNet(model=trained_ElasticNet_model,
#                                 dataloader=outer_train_loader,
#                                 optimizer=optimizer,
#                                 device=device,
#                                 alpha=best_params['alpha'],
#                                 l1_ratio=l1_ratio)

#             # Inspect weights and create node importance
#             # node_importance: the higher the value the more important a feature is
#             # It essentially is a sum of all the wights for a given feature
#             # node_weights = trained_ElasticNet_model.conv1.nn[0].weight.data
#             # node_importance = torch.sum(torch.abs(node_weights), dim=0)
#             node_weights = trained_ElasticNet_model.node_projector.weight.data
#             outer_node_importances.append(torch.sum(torch.abs(node_weights), dim=0).cpu().numpy())

#             # Same for edge features    
#             # edge_weights = trained_ElasticNet_model.conv1.lin.weight.data # This is why we need a custom GINEConv model
#             # edge_importance = torch.sum(torch.abs(edge_weights), dim=0)
#             edge_weights = trained_ElasticNet_model.edge_projector.weight.data
#             outer_edge_importances.append(torch.sum(torch.abs(edge_weights), dim=0).cpu().numpy())

#         # Average the importances to generalize feature importance (not just the max)
#         avg_node_importances = np.mean(outer_node_importances, axis=0)
#         avg_edge_importances = np.mean(outer_edge_importances, axis=0)

#         node_importance_dict = {name: float(importance) for name, importance in zip(node_feature_names, avg_node_importances)}
#         edge_importance_dict = {name: float(importance) for name, importance in zip(edge_feature_names, avg_edge_importances)}

#         feature_selection_importances[prop_name] = {
#             'node_features_importances': node_importance_dict,
#             'edge_features_importances': edge_importance_dict
#         }


#     os.makedirs(output_dir, exist_ok=True) 
#     importances_filename = f'{filename_prefix}_node_edge_features_importances{filename_ending}.json'
#     importances_filepath = output_dir / importances_filename
#     with open(importances_filepath, 'w') as f:
#         json.dump(feature_selection_importances, f, indent=4)
#     print(f"\n----- Computed node & edge features importances -----")
#     print(f"Feature importances saved to '{importances_filepath}'")


def elasticnet_feature_selection(names_filepath,
                                 filename_prefix,
                                 prop_name,
                                 dataset_naming_schema,
                                 dataset_dir,
                                 model_class,
                                 epochs,
                                 hidden_dim_list,
                                 dropout_range,
                                 lr_range,
                                 alpha_range,
                                 device,
                                 n_trials,
                                 outer_folds,
                                 inner_folds,
                                 batch_size,
                                 output_dir,
                                 filename_ending='',
                                 l1_ratio=1.0,
                                 ):
    """
    Performs importance analysis for node & edge features using a given model with
    nested cross-validation. Uses the objective_ElasticNet, an Optuna objective
    function for a single ElasticNet regularization hyperparameter tuning trial,
    to find the optimal hyperparameters.

    The ElasticNet regularization is run for n_trials inside the inner cv fold.

    Outputs a JSON file containing a dictionary where each key is a property name,
    and the value is another dictionary mapping each node & edge feature name to
    its averaged importance score.

    Output schema: {filename_prefix}_node_edge_features_importances_{prop_name}{filename_ending}.json

    Args:
        names_filepath (str): Path to the JSON file containing the list of feature names.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_name (str): The target property name to analyze. e.g., 'Dipole_moment'
        dataset_naming_schema (str): The name for the dataset files. e.g., 'qm9_global_dataset'.
        dataset_dir (str): Path to the directory containing the dataset files.
        model_class (torch.nn.Module): A model class to use for tuning.
        epochs (int): Number of maximum epochs to run ElasticNet training without early stopping.
        patience (int): Number of epochs to wait for improvement before stopping ElasticNet training.
        hidden_dim_list  (List[int]): List of hidden dimensions to try. e.g., [128, 256].
        dropout_range (List[float]): List of the minimum and maximum value to test for the `dropout_rate` hyperparameter.
        lr_range (List[float]): List of the minimum and maximum value to test for the `learning_rate` hyperparameter.
        alpha_range (List[float]): List of the minimum and maximum value to test for the `alpha` ElastiNet hyperparameter.
        device (torch.device): cuda or cpu.
        n_trials (int): The number of Optuna trials to run for each inner loop.
        outer_folds (int): The number of splits for the outer cross-validation loop.
        inner_folds (int): The number of splits for the inner cross-validation loop.
        output_dir (str): Path to the directory to save the JSON file.
        filename_ending (str, optional): The output files suffix.
        l1_ratio (float, optional): The L1 ratio for Elastic Net. Defaults to 1.0 (Lasso).
    """

    try:
        with open(names_filepath, 'r') as f:
            feature_names_data = json.load(f)
        node_feature_names = feature_names_data['node_features_names']
        edge_feature_names = feature_names_data['edge_features_names']
        print(f"Successfully loaded feature names from {names_filepath}")
    except FileNotFoundError:
        print(f"ERROR: Feature name file not found at {names_filepath}")
        print("Run save_feature_names from the data preprocessing notebook.")

    feature_selection_importances = {}

    print(f"\n======= Feature Selection for: {prop_name} =======")

    # Data Loading
    dataset_filename = f'{dataset_naming_schema}_{prop_name}.pt'
    dataset_path = dataset_dir / dataset_filename
    try:
        full_dataset = torch.load(dataset_path, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Run convert_smiles_to_graph from the  preprocessing notebook")

    # Determine feature sizes from the first graph
    num_node_features = full_dataset[0].num_node_features
    num_edge_features = full_dataset[0].num_edge_features

    # Implement nested cross validation
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    outer_node_importances = []
    outer_edge_importances = []

    # ------- Start nested cross-validation -------
    print(" Starting nested cross-validation...")

    for fold_idx, (outer_train_index, outer_test_index) in enumerate(outer_cv.split(full_dataset)):
        print(f'Fold {fold_idx+1} out of {outer_folds}')
        outer_train_dataset = Subset(full_dataset, outer_train_index)


        # Optuna Hyperparameter Study on the outer train_val set
        print(" Running Optuna to find best hyperparameters for this fold...")

        # We define the objective function inside the loop to capture the correct dataset split
        def objective_inner_cv(trial):
            # Suggest Hyperparameters
            hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
            dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
            lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
            alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], log=True)

            val_losses = []

            # Inner loop
            for inner_train_idx, inner_val_idx in inner_cv.split(outer_train_dataset):

                inner_train_dataset = Subset(outer_train_dataset, inner_train_idx)
                inner_val_dataset = Subset(outer_train_dataset, inner_val_idx)

                inner_train_loader = DataLoader(inner_train_dataset, batch_size=batch_size, shuffle=True)
                inner_val_loader = DataLoader(inner_val_dataset, batch_size=batch_size, shuffle=False)

                model = model_class(
                    node_features=num_node_features,
                    edge_features=num_edge_features,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    dropout_rate=dropout_rate
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Train for a smaller, fixed number of epochs for speed
                for _ in range(60):
                    train_ElasticNet(model, inner_train_loader, optimizer, device, alpha, l1_ratio)

                val_loss = test_ElasticNet(model, inner_val_loader, device)
                val_losses.append(val_loss)

            return np.mean(val_losses)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective_inner_cv, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        print(f"Best hyperparameters for {prop_name}: {best_params}")


        # Retrain the model with the best alpha to get the final weights for inspection
        print("\nRetraining with best alpha to get final weights...")

        trained_ElasticNet_model = model_class(
            node_features=num_node_features,
            edge_features=num_edge_features,
            hidden_dim=best_params['hidden_dim'],
            output_dim=1,
            dropout_rate=best_params['dropout_rate']
        ).to(device)
        optimizer = torch.optim.Adam(trained_ElasticNet_model.parameters(), lr=best_params['lr'])

        outer_train_loader = DataLoader(outer_train_dataset, batch_size=batch_size, shuffle=True)
    
        # Train for longer in order to push unimportant params to zero
        for _ in range(epochs):
            train_ElasticNet(model=trained_ElasticNet_model,
                            dataloader=outer_train_loader,
                            optimizer=optimizer,
                            device=device,
                            alpha=best_params['alpha'],
                            l1_ratio=l1_ratio)

        # Inspect weights and create node importance
        # node_importance: the higher the value the more important a feature is
        # It essentially is a sum of all the wights for a given feature
        # node_weights = trained_ElasticNet_model.conv1.nn[0].weight.data
        # node_importance = torch.sum(torch.abs(node_weights), dim=0)
        node_weights = trained_ElasticNet_model.node_projector.weight.data
        outer_node_importances.append(torch.sum(torch.abs(node_weights), dim=0).cpu().numpy())

        # Same for edge features    
        # edge_weights = trained_ElasticNet_model.conv1.lin.weight.data # This is why we need a custom GINEConv model
        # edge_importance = torch.sum(torch.abs(edge_weights), dim=0)
        edge_weights = trained_ElasticNet_model.edge_projector.weight.data
        outer_edge_importances.append(torch.sum(torch.abs(edge_weights), dim=0).cpu().numpy())

    # Average the importances to generalize feature importance (not just the max)
    avg_node_importances = np.mean(outer_node_importances, axis=0)
    avg_edge_importances = np.mean(outer_edge_importances, axis=0)

    node_importance_dict = {name: float(importance) for name, importance in zip(node_feature_names, avg_node_importances)}
    edge_importance_dict = {name: float(importance) for name, importance in zip(edge_feature_names, avg_edge_importances)}

    feature_selection_importances[prop_name] = {
        'node_features_importances': node_importance_dict,
        'edge_features_importances': edge_importance_dict
    }

    os.makedirs(output_dir, exist_ok=True) 
    importances_filename = f'{filename_prefix}_node_edge_features_importances_{prop_name}{filename_ending}.json'
    importances_filepath = output_dir / importances_filename
    with open(importances_filepath, 'w') as f:
        json.dump(feature_selection_importances, f, indent=4)
    print(f"\n----- Computed node & edge features importances fpr {prop_name}-----")
    print(f"Feature importances saved to '{importances_filepath}'")



# def select_node_and_edge_features(names_filepath,
#                                   filename_prefix,
#                                   output_dir,
#                                   threshold_fraction=0.01,
#                                   filename_ending=''
#                                   ):
#     """
#     Selects the most important node & edge features based on the
#     elasticnet_feature_selection function.

#     For each property, it identifies the maximum importance score and applies a
#     threshold based on the provided `threshold_fraction` * max_importance.
#     Features with an importance score above this threshold are considered "selected".

#     threshold_fraction should be between 0 and 1. 

#     The output is a JSON file containing the names and original indices
#     of only the selected features for each property, ready for use in dataset
#     pruning.

#     Output shema: {filename_prefix}_node_edge_features_selection_results{filename_ending}.json
    
#     Args:
#         names_filepath (str): Path to the JSON file containing the list of feature names.
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         threshold_fraction (float): The fraction of the maximum importance score
#             to use as the selection threshold. e.g., 0.01 means features with at
#             least 1% of the max importance will be kept.
#         output_dir (str): Path to the directory to save the JSON file.
#         filename_ending (str, optional): The output files suffix.
#     """
    
#     feature_selection_results = {}
#     node_edge_feature_selection_importances = {}

#     try:
#         with open(names_filepath, 'r') as f:
#             feature_names_data = json.load(f)
#         node_feature_names = feature_names_data['node_features_names']
#         edge_feature_names = feature_names_data['edge_features_names']
#         print(f"Successfully loaded feature names from {names_filepath}")
#     except FileNotFoundError:
#         print(f"ERROR: Feature name file not found at {names_filepath}")
#         print("Run save_feature_names from the data preprocessing notebook.")

#     importances_filename = f'{filename_prefix}_node_edge_features_importances{filename_ending}.json'
#     importances_filepath = output_dir / importances_filename
#     try:
#         with open(importances_filepath, 'r') as f:
#             node_edge_feature_selection_importances = json.load(f)
#         print(f"Loaded importances from '{importances_filepath}'")
#     except FileNotFoundError:
#         print(f"ERROR: Feature importance file not found at {importances_filepath}")
#         print(f"Run savefeatureimporatces models analysis notebook")


#     for prop_name, data_dict in node_edge_feature_selection_importances.items():
#         print(f"\n======= Node & edge feature selection for: {prop_name} =======")

#         node_importance_dict = data_dict['node_features_importances']
#         edge_importance_dict = data_dict['edge_features_importances']

#         node_importance_values = np.array(list(node_importance_dict.values()))
#         edge_importance_values = np.array(list(edge_importance_dict.values()))

#         node_max_importance = np.max(node_importance_values)
#         edge_max_importance = np.max(edge_importance_values)

#         # threshold_fraction = 0.01 # We just want to eliminate values close to zero

#         node_threshold = threshold_fraction * node_max_importance
#         edge_threshold = threshold_fraction * edge_max_importance

#         node_selected_indices = [i for i, importance in enumerate(node_importance_values) if importance >= node_threshold]
#         node_selected_names = [node_feature_names[i] for i in node_selected_indices]

#         edge_selected_indices = [i for i, importance in enumerate(edge_importance_values) if importance >= edge_threshold]
#         edge_selected_names = [edge_feature_names[i] for i in edge_selected_indices]

#         feature_selection_results[prop_name] = {
#             'node_features_index': node_selected_indices,
#             'node_features_selected_names': node_selected_names,
#             'edge_features_index': edge_selected_indices,
#             'edge_features_selected_names': edge_selected_names
#         }


#         print(f"\nNode & edge feature importance for {prop_name}")
#         print(f"Total node features selected: {len(node_selected_indices)} out of {len(node_feature_names)}")
#         print(f"Total edge features selected: {len(edge_selected_indices)} out of {len(edge_feature_names)}")
#         print("\nTop 10 node features:")
#         sorted_indices = np.argsort(node_importance_values)[::-1]
#         for i in sorted_indices[:10]:
#             print(f"       - {node_feature_names[i]}: {node_importance_values[i]:.4f}")
#         print("\nTop 10 edge features:")
#         sorted_indices = np.argsort(edge_importance_values)[::-1]
#         for i in sorted_indices[:10]:
#             print(f"       - {edge_feature_names[i]}: {edge_importance_values[i]:.4f}")

#     # Save the results
#     os.makedirs(output_dir, exist_ok=True) 
#     selection_filename = f'{filename_prefix}_node_edge_features_selection_results{filename_ending}.json'
#     selection_filepath = output_dir / selection_filename
#     with open(selection_filepath, 'w') as f:
#         json.dump(feature_selection_results, f, indent=4)

#     print(f"\nFeature selection results saved to '{selection_filepath}'")


def select_node_and_edge_features(names_filepath,
                                  filename_prefix,
                                  prop_name,
                                  output_dir,
                                  threshold_fraction=0.01,
                                  filename_ending=''
                                  ):
    """
    Selects the most important node & edge features based on the
    elasticnet_feature_selection function.

    For each property, it identifies the maximum importance score and applies a
    threshold based on the provided `threshold_fraction` * max_importance.
    Features with an importance score above this threshold are considered "selected".

    threshold_fraction should be between 0 and 1. 

    The output is a JSON file containing the names and original indices
    of only the selected features for each property, ready for use in dataset
    pruning.

    Output shema: {filename_prefix}_node_edge_features_selection_{prop_name}{filename_ending}.json
    
    Args:
        names_filepath (str): Path to the JSON file containing the list of feature names.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_name (str): The target property name to analyze. e.g., 'Dipole_moment'
        threshold_fraction (float): The fraction of the maximum importance score
            to use as the selection threshold. e.g., 0.01 means features with at
            least 1% of the max importance will be kept.
        output_dir (str): Path to the directory to save the JSON file.
        filename_ending (str, optional): The output files suffix.
    """
    
    feature_selection_results = {}
    node_edge_feature_selection_importances = {}

    try:
        with open(names_filepath, 'r') as f:
            feature_names_data = json.load(f)
        node_feature_names = feature_names_data['node_features_names']
        edge_feature_names = feature_names_data['edge_features_names']
        print(f"Successfully loaded feature names from {names_filepath}")
    except FileNotFoundError:
        print(f"ERROR: Feature name file not found at {names_filepath}")
        print("Run save_feature_names from the data preprocessing notebook.")

    importances_filename = f'{filename_prefix}_node_edge_features_importances_{prop_name}{filename_ending}.json'
    importances_filepath = output_dir / importances_filename
    try:
        with open(importances_filepath, 'r') as f:
            node_edge_feature_selection_importances = json.load(f)
        print(f"Loaded importances from '{importances_filepath}'")
    except FileNotFoundError:
        print(f"ERROR: Feature importance file not found at {importances_filepath}")
        print(f"Run savefeatureimporatces models analysis notebook")


    print(f"\n======= Node & edge feature selection for: {prop_name} =======")

    data_dict = node_edge_feature_selection_importances[prop_name]
    node_importance_dict = data_dict['node_features_importances']
    edge_importance_dict = data_dict['edge_features_importances']

    node_importance_values = np.array(list(node_importance_dict.values()))
    edge_importance_values = np.array(list(edge_importance_dict.values()))

    node_max_importance = np.max(node_importance_values)
    edge_max_importance = np.max(edge_importance_values)

    # threshold_fraction = 0.01 # We just want to eliminate values close to zero

    node_threshold = threshold_fraction * node_max_importance
    edge_threshold = threshold_fraction * edge_max_importance

    node_selected_indices = [i for i, importance in enumerate(node_importance_values) if importance >= node_threshold]
    node_selected_names = [node_feature_names[i] for i in node_selected_indices]

    edge_selected_indices = [i for i, importance in enumerate(edge_importance_values) if importance >= edge_threshold]
    edge_selected_names = [edge_feature_names[i] for i in edge_selected_indices]

    feature_selection_results[prop_name] = {
        'node_features_index': node_selected_indices,
        'node_features_selected_names': node_selected_names,
        'edge_features_index': edge_selected_indices,
        'edge_features_selected_names': edge_selected_names
    }


    print(f"\nNode & edge feature importance for {prop_name}")
    print(f"Total node features selected: {len(node_selected_indices)} out of {len(node_feature_names)}")
    print(f"Total edge features selected: {len(edge_selected_indices)} out of {len(edge_feature_names)}")
    print("\nTop 10 node features:")
    sorted_indices = np.argsort(node_importance_values)[::-1]
    for i in sorted_indices[:10]:
        print(f"       - {node_feature_names[i]}: {node_importance_values[i]:.4f}")
    print("\nTop 10 edge features:")
    sorted_indices = np.argsort(edge_importance_values)[::-1]
    for i in sorted_indices[:10]:
        print(f"       - {edge_feature_names[i]}: {edge_importance_values[i]:.4f}")

    # Save the results
    os.makedirs(output_dir, exist_ok=True) 
    selection_filename = f'{filename_prefix}_node_edge_features_selection_{prop_name}{filename_ending}.json'
    selection_filepath = output_dir / selection_filename
    with open(selection_filepath, 'w') as f:
        json.dump(feature_selection_results, f, indent=4)

    print(f"\nFeature selection results for {prop_name} saved to '{selection_filepath}'")


# ===== Create final dataset using selected features =====


# def prune_dataset(global_features_filepath,
#                   filename_prefix,
#                   prop_names,
#                   json_dir,
#                   dataset_dir,
#                   output_dir,
#                   filename_ending='',
#                   ):
#     """
#     Creates the final pruned dataset using only the selected features.

#     Output schema: {filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt

#     Args:
#         global_features_filepath (str): Path to the JSON file containing all precalculated global features.
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
#         json_dir (str): Path to the directory of feature selection results.
#         dataset_dir (str): Path to the directory containing the dataset file.
#         output_dir (str): Path to the directory to save the .pt file.
#         filename_ending (str, optional): The output files suffix.
#     """

#     node_edge_selection_filename = f'{filename_prefix}_node_edge_features_selection_results{filename_ending}.json'
#     node_edge_selection_filepath = json_dir / node_edge_selection_filename
#     try:
#         with open(node_edge_selection_filepath, 'r') as f:
#             node_edge_selection = json.load(f)
#         print(f"Successfully loaded node & edge feature selection from {node_edge_selection_filepath}")
#     except FileNotFoundError:
#         print(f"ERROR: File not found at {node_edge_selection_filepath}")
#         print("Ensure the node & edge feature selection has been run first.")


#     global_selection_filename = f'{filename_prefix}_global_features_selection_results{filename_ending}.json'
#     global_selection_filepath = json_dir / global_selection_filename
#     try:
#         with open(global_selection_filepath, 'r') as f:
#             global_selection = json.load(f)
#         print(f"Successfully loaded global feature selection from {global_selection_filepath}")
#     except FileNotFoundError:
#         print(f"ERROR: File not found at {global_selection_filepath}")
#         print("Ensure the global feature selection has been run first.")


#     try:
#         print(f"\nLoading all global features values...")
#         with open(global_features_filepath, 'r') as f:
#             global_features = json.load(f)
#         print(f"Successfully loaded all global features from {global_features_filepath}")
#     except FileNotFoundError:
#         print(f"ERROR: File not found at {global_features_filepath}")
#         print("Ensure the save_global_features function from the preprocess notebook has been run first.")


#     for prop_name in prop_names:
#         print(f"\n======= Pruning dataset for: {prop_name} =======")

#         # Load the original dataset
#         dataset_filename = f'{filename_prefix}_dataset_{prop_name}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             local_features_dataset = torch.load(dataset_path, weights_only=False)
#         except FileNotFoundError:
#             print(f"ERROR: Original {filename_prefix}_dataset_{prop_name} not found at {dataset_path}")
#             print("Ensure the data preprocessing notebook has been run first.")

#         # Index of features to keep
#         node_indices = node_edge_selection.get(prop_name, {}).get('node_features_index', list([1, 2]))
#         edge_indices = node_edge_selection.get(prop_name, {}).get('edge_features_index', list([1, 2]))
#         global_indices = global_selection.get(prop_name, {}).get('global_features_index', list([1, 2]))

#         print(f"Keeping {len(node_indices)} node, {len(edge_indices)} edge, and {len(global_indices)} global features.")

#         # Create the pruned dataset
#         pruned_dataset = []
#         for i, data in enumerate(tqdm(local_features_dataset, desc=f"Pruning {prop_name}")):

#             pruned_data = data.clone()
#             pruned_data.x = data.x[:, node_indices]
#             pruned_data.edge_attr = data.edge_attr[:, edge_indices]

#             global_features_list = global_features[i]
#             pruned_global_feats = [global_features_list[j] for j in global_indices]
#             pruned_global_feats = [0.0 if np.isnan(v) else v for v in pruned_global_feats] # Handle NaNs
#             pruned_data.u = torch.tensor(pruned_global_feats, dtype=torch.float).view(1, -1)

#             pruned_dataset.append(pruned_data)

#         # Save the pruned dataset
#         os.makedirs(output_dir, exist_ok=True) 
#         pruned_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
#         pruned_filepath = output_dir / pruned_filename
#         torch.save(pruned_dataset, pruned_filepath)
#         print(f" Successfully created and saved pruned dataset to '{pruned_filepath}'")

#     print("\n----- All datasets have been pruned and saved. -----")

# def prune_dataset(global_features_filepath,
#                   filename_prefix,
#                   prop_names,
#                   json_dir,
#                   dataset_dir,
#                   output_dir,
#                   filename_ending='',
#                   ):
#     """
#     Creates the final pruned dataset using only the selected features.

#     Output schema: {filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt

#     Args:
#         global_features_filepath (str): Path to the JSON file containing all precalculated global features.
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
#         json_dir (str): Path to the directory of feature selection results.
#         dataset_dir (str): Path to the directory containing the dataset file.
#         output_dir (str): Path to the directory to save the .pt file.
#         filename_ending (str, optional): The output files suffix.
#     """

#     for prop_name in prop_names:
#         print(f"\n======= Pruning dataset for: {prop_name} =======")

#         node_edge_selection_filename = f'{filename_prefix}_node_edge_features_selection_results_{prop_name}{filename_ending}.json'
#         node_edge_selection_filepath = json_dir / node_edge_selection_filename
#         try:
#             with open(node_edge_selection_filepath, 'r') as f:
#                 node_edge_selection = json.load(f)
#             print(f"Successfully loaded node & edge feature selection from {node_edge_selection_filepath}")
#         except FileNotFoundError:
#             print(f"ERROR: File not found at {node_edge_selection_filepath}")
#             print("Ensure the node & edge feature selection has been run first.")


#         global_selection_filename = f'{filename_prefix}_global_features_selection_results_{prop_name}{filename_ending}.json'
#         global_selection_filepath = json_dir / global_selection_filename
#         try:
#             with open(global_selection_filepath, 'r') as f:
#                 global_selection = json.load(f)
#             print(f"Successfully loaded global feature selection from {global_selection_filepath}")
#         except FileNotFoundError:
#             print(f"ERROR: File not found at {global_selection_filepath}")
#             print("Ensure the global feature selection has been run first.")


#         try:
#             print(f"\nLoading all global features values...")
#             with open(global_features_filepath, 'r') as f:
#                 global_features = json.load(f)
#             print(f"Successfully loaded all global features from {global_features_filepath}")
#         except FileNotFoundError:
#             print(f"ERROR: File not found at {global_features_filepath}")
#             print("Ensure the save_global_features function from the preprocess notebook has been run first.")

#         # Load the original dataset
#         dataset_filename = f'{filename_prefix}_dataset_{prop_name}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             local_features_dataset = torch.load(dataset_path, weights_only=False)
#         except FileNotFoundError:
#             print(f"ERROR: Original {filename_prefix}_dataset_{prop_name} not found at {dataset_path}")
#             print("Ensure the data preprocessing notebook has been run first.")

#         # Index of features to keep
#         node_indices = node_edge_selection.get(prop_name, {}).get('node_features_index', [])
#         edge_indices = node_edge_selection.get(prop_name, {}).get('edge_features_index', [])
#         global_indices = global_selection.get(prop_name, {}).get('global_features_index', [])

#         print(f"Keeping {len(node_indices)} node, {len(edge_indices)} edge, and {len(global_indices)} global features.")

#         # Create the pruned dataset
#         pruned_dataset = []
#         for i, data in enumerate(tqdm(local_features_dataset, desc=f"Pruning {prop_name}")):

#             pruned_data = data.clone()
#             pruned_data.x = data.x[:, node_indices]
#             pruned_data.edge_attr = data.edge_attr[:, edge_indices]

#             global_features_list = global_features[i]
#             pruned_global_feats = [global_features_list[j] for j in global_indices]
#             pruned_global_feats = [0.0 if np.isnan(v) else v for v in pruned_global_feats] # Handle NaNs
#             pruned_data.u = torch.tensor(pruned_global_feats, dtype=torch.float).view(1, -1)

#             pruned_dataset.append(pruned_data)

#         # Save the pruned dataset
#         os.makedirs(output_dir, exist_ok=True) 
#         pruned_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
#         pruned_filepath = output_dir / pruned_filename
#         torch.save(pruned_dataset, pruned_filepath)
#         print(f" Successfully created and saved pruned dataset to '{pruned_filepath}'")

#     print("\n----- All datasets have been pruned and saved. -----")


def prune_dataset(global_features_filepath,
                  filename_prefix,
                  prop_name,
                  json_dir,
                  dataset_dir,
                  output_dir,
                  filename_ending='',
                  ):
    """
    Creates the final pruned dataset using only the selected features.

    Output schema: {filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt

    Args:
        global_features_filepath (str): Path to the JSON file containing all precalculated global features.
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_name (str): The target property name to analyze. e.g., 'Dipole_moment'
        json_dir (str): Path to the directory of feature selection results.
        dataset_dir (str): Path to the directory containing the dataset file.
        output_dir (str): Path to the directory to save the .pt file.
        filename_ending (str, optional): The output files suffix.
    """


    print(f"\n======= Pruning dataset for: {prop_name} =======")

    node_edge_selection_filename = f'{filename_prefix}_node_edge_features_selection_{prop_name}{filename_ending}.json'
    node_edge_selection_filepath = json_dir / node_edge_selection_filename
    try:
        with open(node_edge_selection_filepath, 'r') as f:
            node_edge_selection = json.load(f)
        print(f"Successfully loaded node & edge feature selection from {node_edge_selection_filepath}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {node_edge_selection_filepath}")
        print("Ensure the node & edge feature selection has been run first.")


    global_selection_filename = f'{filename_prefix}_global_features_selection_{prop_name}{filename_ending}.json'
    global_selection_filepath = json_dir / global_selection_filename
    try:
        with open(global_selection_filepath, 'r') as f:
            global_selection = json.load(f)
        print(f"Successfully loaded global feature selection from {global_selection_filepath}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {global_selection_filepath}")
        print("Ensure the global feature selection has been run first.")


    try:
        print(f"\nLoading all global features values...")
        with open(global_features_filepath, 'r') as f:
            global_features = json.load(f)
        print(f"Successfully loaded all global features from {global_features_filepath}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {global_features_filepath}")
        print("Ensure the save_global_features function from the preprocess notebook has been run first.")

    # Load the original dataset
    dataset_filename = f'{filename_prefix}_dataset_{prop_name}.pt'
    dataset_path = dataset_dir / dataset_filename
    try:
        local_features_dataset = torch.load(dataset_path, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Original {filename_prefix}_dataset_{prop_name} not found at {dataset_path}")
        print("Ensure the data preprocessing notebook has been run first.")

    # Index of features to keep
    node_indices = node_edge_selection.get(prop_name, {}).get('node_features_index', [])
    edge_indices = node_edge_selection.get(prop_name, {}).get('edge_features_index', [])
    global_indices = global_selection.get(prop_name, {}).get('global_features_index', [])

    print(f"Keeping {len(node_indices)} node, {len(edge_indices)} edge, and {len(global_indices)} global features.")

    # Create the pruned dataset
    pruned_dataset = []
    for i, data in enumerate(tqdm(local_features_dataset, desc=f"Pruning {prop_name}")):

        pruned_data = data.clone()
        pruned_data.x = data.x[:, node_indices]
        pruned_data.edge_attr = data.edge_attr[:, edge_indices]

        global_features_list = global_features[i]
        pruned_global_feats = [global_features_list[j] for j in global_indices]
        pruned_global_feats = [0.0 if np.isnan(v) else v for v in pruned_global_feats] # Handle NaNs
        pruned_data.u = torch.tensor(pruned_global_feats, dtype=torch.float).view(1, -1)

        pruned_dataset.append(pruned_data)

    # Save the pruned dataset
    os.makedirs(output_dir, exist_ok=True) 
    pruned_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
    pruned_filepath = output_dir / pruned_filename
    torch.save(pruned_dataset, pruned_filepath)
    print(f" Successfully created and saved pruned dataset to '{pruned_filepath}'")



# ===== Hyperparameter tuning =====

# def hyperparameter_tuning(filename_prefix,
#                           prop_names,
#                           index_dir,
#                           dataset_dir,
#                           batch_size,
#                           model_class,
#                           device,
#                           epochs,
#                           hidden_dim_list,
#                           dropout_range,
#                           lr_range,
#                           patience,
#                           n_trials,
#                           output_dir,
#                           filename_ending=''
#                           ):
#     """
#     Performs hyperparameter optimization for a given model using optuna.

#     Uses the training and validation splits from split_data_to_train_val_test.

#     Uses the objective_with_globals, an Optuna objective function for a single
#     hyperparameter tuning trial, to find the optimal hyperparameters. Each trial
#     is scored by its best validation loss. Adds support for models with global
#     features.

#     Outputs a JSON file containing the best set of hyperparameters for
#     each property.

#     Output schema: {filename_prefix}_best_hyperparameters{filename_ending}.json

#     Args:
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
#         index_dir (str): Path to the directory containing the index files.
#         dataset_dir (str): Path to the directory containing the dataset files.
#         model_class (torch.nn.Module): A model class to use for tuning.
#         device (torch.device): cuda or cpu.
#         epochs (int): Number of maximum epochs to run each trial.
#         hidden_dim_list  (List[int]): List of hidden dimensions to try. e.g., [128, 256].
#         dropout_range (List[float]): List of the minimum and maximum value to test for the `dropout_rate` hyperparameter.
#         lr_range (List[float]): List of the minimum and maximum value to test for the `learning_rate` hyperparameter.
#         patience (int): Number of epochs to wait for improvement before stopping hyperparameter tuning.
#         n_trials (int): The number of Optuna trials to run for each property.
#         output_dir (str): Path to the directory to save the JSON file.
#         filename_ending (str, optional): The output files suffix.
#     """
    
#     # Load indexes from CSV files
#     print("Loading train, validation, test indexes from CSV...")
#     train_path = index_dir / f'{filename_prefix}_index_train{filename_ending}.csv'
#     val_path = index_dir / f'{filename_prefix}_index_validation{filename_ending}.csv'

#     index_train_df = pd.read_csv(train_path)
#     index_val_df = pd.read_csv(val_path)

#     # Convert the DataFrame columns to NumPy arrays (or lists) for indexing
#     train_index = index_train_df['index'].values
#     val_index = index_val_df['index'].values

#     print(f"Train samples: {len(train_index)}, Validation samples: {len(val_index)}")

#     # fold_best_params = {}
#     best_params = {}
#     # fold_best_loss = {}
#     # best_loss = {}
#     # best_trial = {}

#     for prop_name in prop_names:
#         print(f"\n======= Hyperparameter tuning for: {prop_name} =======")

#         # Data Loading
#         dataset_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             full_dataset = torch.load(dataset_path, weights_only=False)
#             print(f"Loaded {prop_name} dataset")
#         except FileNotFoundError:
#                 print(f"ERROR: Dataset not found at {dataset_path}")
#                 print("Ensure the dataset pruning cell has been run first.")

#         # Determine feature sizes from the first graph
#         num_node_features = full_dataset[0].x.shape[1]
#         num_edge_features = full_dataset[0].edge_attr.shape[1]
#         num_global_features = full_dataset[0].u.shape[1]

#         # Split train and validation sets
#         train_dataset = Subset(full_dataset, train_index)
#         val_dataset = Subset(full_dataset, val_index)

#         # Create DataLoaders
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#         # --- Create and Run the Optuna Study ---
#         study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
#         study.optimize(
#             lambda trial: objective_with_globals(trial,
#                                                 model_class,
#                                                 train_loader,
#                                                 val_loader,
#                                                 num_node_features,
#                                                 num_edge_features,
#                                                 num_global_features,
#                                                 epochs,
#                                                 hidden_dim_list,
#                                                 dropout_range,
#                                                 lr_range,
#                                                 device=device,
#                                                 patience=patience),                                   
#             n_trials=n_trials, show_progress_bar=True  # The number of different hyperparameter combinations to try
#         )

#         best_params[prop_name] = study.best_params

#         print(f"\n----- Finished Tuning for {prop_name} -----")
#         print(f"Best trial number: {study.best_trial.number}")
#         print(f"Best validation MSE: {study.best_value}")
#         print(f"Best hyperparameters: {study.best_params}")

#     print("\n\n======= FINAL OPTIMAL HYPERPARAMETERS =======")
#     for prop_name, params in best_params.items():
#         print(f"\n--- {prop_name} ---")
#         for param_name, param_value in params.items():
#             print(f"  {param_name}: {param_value}")

#     # Save in json
#     os.makedirs(output_dir, exist_ok=True) 
#     params_filename = f'{filename_prefix}_best_hyperparameters{filename_ending}.json'
#     params_filepath = output_dir / params_filename
#     with open(params_filepath, 'w') as f:
#         json.dump(best_params, f, indent=4)

#     print(f"\nOptimal hyperparameters saved to: {params_filepath}")


def hyperparameter_tuning(filename_prefix,
                          prop_name,
                          index_dir,
                          dataset_dir,
                          batch_size,
                          model_class,
                          device,
                          epochs,
                          hidden_dim_list,
                          dropout_range,
                          lr_range,
                          patience,
                          n_trials,
                          output_dir,
                          filename_ending=''
                          ):
    """
    Performs hyperparameter optimization for a given model using optuna.

    Uses the training and validation splits from split_data_to_train_val_test.

    Uses the objective_with_globals, an Optuna objective function for a single
    hyperparameter tuning trial, to find the optimal hyperparameters. Each trial
    is scored by its best validation loss. Adds support for models with global
    features.

    Outputs a JSON file containing the best set of hyperparameters for
    each property.

    Output schema: {filename_prefix}_best_hyperparameters_{prop_name}{filename_ending}.json

    Args:
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_name (str): The target property name to analyze. e.g., 'Dipole_moment'
        index_dir (str): Path to the directory containing the index files.
        dataset_dir (str): Path to the directory containing the dataset files.
        model_class (torch.nn.Module): A model class to use for tuning.
        device (torch.device): cuda or cpu.
        epochs (int): Number of maximum epochs to run each trial.
        hidden_dim_list  (List[int]): List of hidden dimensions to try. e.g., [128, 256].
        dropout_range (List[float]): List of the minimum and maximum value to test for the `dropout_rate` hyperparameter.
        lr_range (List[float]): List of the minimum and maximum value to test for the `learning_rate` hyperparameter.
        patience (int): Number of epochs to wait for improvement before stopping hyperparameter tuning.
        n_trials (int): The number of Optuna trials to run for each property.
        output_dir (str): Path to the directory to save the JSON file.
        filename_ending (str, optional): The output files suffix.
    """
    
    # Load indexes from CSV files
    print("Loading train, validation, test indexes from CSV...")
    train_path = index_dir / f'{filename_prefix}_index_train{filename_ending}.csv'
    val_path = index_dir / f'{filename_prefix}_index_validation{filename_ending}.csv'

    index_train_df = pd.read_csv(train_path)
    index_val_df = pd.read_csv(val_path)

    # Convert the DataFrame columns to NumPy arrays (or lists) for indexing
    train_index = index_train_df['index'].values
    val_index = index_val_df['index'].values

    print(f"Train samples: {len(train_index)}, Validation samples: {len(val_index)}")

    best_params = {}

    print(f"\n======= Hyperparameter tuning for: {prop_name} =======")

    # Data Loading
    dataset_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
    dataset_path = dataset_dir / dataset_filename
    try:
        full_dataset = torch.load(dataset_path, weights_only=False)
        print(f"Loaded {prop_name} dataset")
    except FileNotFoundError:
            print(f"ERROR: Dataset not found at {dataset_path}")
            print("Ensure the dataset pruning cell has been run first.")

    # Determine feature sizes from the first graph
    num_node_features = full_dataset[0].x.shape[1]
    num_edge_features = full_dataset[0].edge_attr.shape[1]
    num_global_features = full_dataset[0].u.shape[1]

    # Split train and validation sets
    train_dataset = Subset(full_dataset, train_index)
    val_dataset = Subset(full_dataset, val_index)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Create and Run the Optuna Study ---
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective_with_globals(trial,
                                            model_class,
                                            train_loader,
                                            val_loader,
                                            num_node_features,
                                            num_edge_features,
                                            num_global_features,
                                            epochs,
                                            hidden_dim_list,
                                            dropout_range,
                                            lr_range,
                                            device=device,
                                            patience=patience),                                   
        n_trials=n_trials, show_progress_bar=True  # The number of different hyperparameter combinations to try
    )

    best_params = {prop_name: study.best_params}

    print(f"\n----- Finished Tuning for {prop_name} -----")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best validation MSE: {study.best_value}")
    print(f"Best hyperparameters: {study.best_params}")

    print("\n\n======= FINAL OPTIMAL HYPERPARAMETERS =======")
    for prop_name, params in best_params.items():
        print(f"\n--- {prop_name} ---")
        for param_name, param_value in params.items():
            print(f"  {param_name}: {param_value}")

    # Save in json
    os.makedirs(output_dir, exist_ok=True) 
    params_filename = f'{filename_prefix}_best_hyperparameters_{prop_name}{filename_ending}.json'
    params_filepath = output_dir / params_filename
    with open(params_filepath, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\nOptimal hyperparameters for {prop_name} saved to: {params_filepath}")



# ===== Nested cross validation =====


# def nested_cross_validation(filename_prefix,
#                             prop_names,
#                             model_naming_schema,
#                             models_dir,
#                             dataset_dir,
#                             scalers,
#                             model_class,
#                             epochs,
#                             hidden_dim_list,
#                             dropout_range,
#                             lr_range,
#                             device,
#                             n_trials,
#                             outer_folds,
#                             inner_folds,
#                             batch_size,
#                             output_dir,
#                             filename_ending='',
#                             patience = 10
#                             ):
#     """
#     Performs importance analysis for node & edge features using a given model with
#     nested cross-validation. Uses the objective_ElasticNet, an Optuna objective
#     function for a single ElasticNet regularization hyperparameter tuning trial,
#     to find the optimal hyperparameters.

#     The ElasticNet regularization is run for n_trials inside the inner cv fold.

#     Outputs a JSON file containing a dictionary where each key is a property name,
#     and the value is another dictionary mapping each node & edge feature name to
#     its averaged importance score.

#     Output schema: {filename_prefix}_nested_cv_scores{filename_ending}.json
#                    {filename_prefix}_nested_cv_predictions{filename_ending}.json
#                    {filename_prefix}_nested_cv_hyperparameters{filename_ending}.json

#     Args:
#         filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
#         prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
#         model_naming_schema (str): The name for saving the models e.g., 'GINE_globals'.
#         models_dir (str): Path to the directory to save the files.
#         dataset_dir (str): Path to the directory containing the dataset files.
#         scalers (dictionary): Dictionary with keys property names and values path to scalers
#         model_class (torch.nn.Module): A model class to use for tuning.
#         epochs (int): Number of maximum epochs to run training for that fold.
#         patience (int): Number of epochs to wait for improvement before stopping training.
#         hidden_dim_list  (List[int]): List of hidden dimensions to try. e.g., [128, 256].
#         dropout_range (List[int]): List of the minimum and maximum value to test for the `dropout_rate` hyperparameter.
#         lr_range (List[float]): List of the minimum and maximum value to test for the `learning_rate` hyperparameter.
#         device (torch.device): cuda or cpu.
#         n_trials (int): The number of Optuna trials to run for each inner loop.
#         outer_folds (int): The number of splits for the outer cross-validation loop.
#         inner_folds (int): The number of splits for the inner cross-validation loop.
#         batch_size (int): Number of molecules per training batch. More is faster but requires more vram.
#         output_dir (str): Path to the directory to save the JSON file.
#         filename_ending (str, optional): The output files suffix.
#         l1_ratio (float, optional): The L1 ratio for Elastic Net. Defaults to 1.0 (Lasso).
#     """

#     # This will store the final test scores from each outer fold
#     final_test_scores = {prop: [] for prop in prop_names}
#     final_fold_predictions = {prop: [] for prop in prop_names}
#     final_fold_best_params = {prop: [] for prop in prop_names}


#     for prop_name in prop_names:
#         print(f"\n======= Nested CV for: {prop_name} =======")

#         # Data Loading
#         dataset_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             full_dataset = torch.load(dataset_path, weights_only=False)
#             print(f"Loaded {prop_name} dataset")
#         except FileNotFoundError:
#                 print(f"ERROR: Dataset not found at {dataset_path}")
#                 print("Ensure the dataset pruning cell has been run first.")

#         # Determine feature sizes from the first graph
#         num_node_features = full_dataset[0].x.shape[1]
#         num_edge_features = full_dataset[0].edge_attr.shape[1]
#         num_global_features = full_dataset[0].u.shape[1]

#         # Implement nested cross validation
#         outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
#         inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

#         # ------- Start nested cross-validation -------
#         print(" Starting nested cross-validation...")

#         for fold_idx, (outer_train_index, outer_test_index) in enumerate(outer_cv.split(full_dataset)):
#             print(f'Fold {fold_idx+1} out of {outer_folds}')

#             # Create DataLoaders
#             outer_train_dataset = Subset(full_dataset, outer_train_index)
#             outer_test_dataset = Subset(full_dataset, outer_test_index)
#             outer_test_loader = DataLoader(outer_test_dataset, batch_size=batch_size, shuffle=False)

#             # Optuna Hyperparameter Study on the outer train_val set
#             print(" Running Optuna to find best hyperparameters for this fold...")

#             # We define the objective function inside the loop to capture the correct dataset split
#             def objective_inner_cv(trial):
#                 # Suggest Hyperparameters
#                 hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
#                 dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
#                 lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)

#                 val_losses = []

#                 # Inner loop
#                 for inner_train_idx, inner_val_idx in inner_cv.split(outer_train_dataset):
#                     inner_train_dataset = Subset(outer_train_dataset, inner_train_idx)
#                     inner_val_dataset = Subset(outer_train_dataset, inner_val_idx)
#                     inner_train_loader = DataLoader(inner_train_dataset, batch_size=batch_size, shuffle=True)
#                     inner_val_loader = DataLoader(inner_val_dataset, batch_size=batch_size, shuffle=False)

#                     model = model_class(
#                         node_features=num_node_features,
#                         edge_features=num_edge_features,
#                         global_features=num_global_features,
#                         hidden_dim=hidden_dim,
#                         output_dim=1,
#                         dropout_rate=dropout_rate
#                     ).to(device)
#                     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#                     loss_fn = nn.MSELoss()

#                     # Train for a smaller, fixed number of epochs for speed
#                     for _ in range(60):
#                         train(model, inner_train_loader, optimizer, loss_fn, device)

#                     val_loss = test(model, inner_val_loader, loss_fn, device)
#                     val_losses.append(val_loss)

#                 return np.mean(val_losses)

#             study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
#             study.optimize(objective_inner_cv, n_trials=n_trials, show_progress_bar=True)

#             best_params = study.best_params
#             print(f"Best hyperparameters: {best_params}")

#             # Train the final model on the outer training set
#             model_name = f"{model_naming_schema}_{prop_name}_fold_{fold_idx+1}"

#             # Instantiate the final model for training
#             final_fold_model = model_class(
#                 node_features=num_node_features,
#                 edge_features=num_edge_features,
#                 global_features=num_global_features,
#                 hidden_dim=best_params['hidden_dim'],
#                 output_dim=1,
#                 dropout_rate=best_params['dropout_rate']
#             ).to(device)

#             outer_train_loader = DataLoader(outer_train_dataset, batch_size=batch_size, shuffle=True)

#             trained_model, _ = train_with_early_stopping(
#                 final_fold_model,
#                 model_name,
#                 models_dir,
#                 filename_prefix,
#                 outer_train_loader,
#                 outer_test_loader,
#                 epochs=epochs,
#                 lr=best_params['lr'],
#                 device=device,
#                 patience=patience,
#                 filename_ending=filename_ending)

#             # Evaluate on the outer test set
#             print(" Evaluating on outer test fold...")
#             trained_model.eval()
#             all_preds = []
#             all_true = []
#             with torch.no_grad():
#                 for data in outer_test_loader:
#                     data = data.to(device)
#                     all_preds.append(trained_model(data).cpu().numpy())
#                     all_true.append(data.y.cpu().numpy())

#             preds = np.concatenate(all_preds, axis=0)
#             true = np.concatenate(all_true, axis=0)

#             scaler = scalers[prop_name]
#             unscaled_preds = scaler.inverse_transform(preds)
#             unscaled_true = scaler.inverse_transform(true)

#             mse = mean_squared_error(unscaled_true, unscaled_preds)
#             mae = mean_absolute_error(unscaled_true, unscaled_preds)
#             rmse = root_mean_squared_error(unscaled_true, unscaled_preds)
#             r2 = r2_score(unscaled_true, unscaled_preds)

#             print(f"\n Fold Test MSE: {mse:.5f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, R2: {r2:.5f}")

#             scores_data = {
#                 'fold_index': fold_idx,
#                 'MSE': float(mse),
#                 'MAE': float(mae),
#                 'RMSE': float(rmse),
#                 'R2': float(r2)
#             }
#             final_test_scores[prop_name].append(scores_data)

#             fold_prediction_data = {
#                 'fold_index': fold_idx,
#                 'predictions': unscaled_preds.tolist(),
#                 'true_values': unscaled_true.tolist()
#             }

#             final_fold_predictions[prop_name].append(fold_prediction_data)

#             params_data = best_params.copy() # Make a copy
#             params_data['fold_index'] = fold_idx
#             final_fold_best_params[prop_name].append(params_data)

#             final_fold_best_params[prop_name].append(best_params)

#     # Report results
#     print("\n\n======= Nested cross validation results (Mean +/- Std dev across folds) =======")
#     for prop_name, scores_list in final_test_scores.items():
#         mse_scores = [s['MSE'] for s in scores_list]
#         mae_scores = [s['MAE'] for s in scores_list]
#         rmse_scores = [s['RMSE'] for s in scores_list]
#         r2_scores = [s['R2'] for s in scores_list]

#         # Calculate mean and standard deviation
#         mean_mse, std_mse = np.mean(mse_scores), np.std(mse_scores)
#         mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
#         mean_rmse, std_rmse = np.mean(rmse_scores), np.std(rmse_scores)
#         mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)

#         print(f"\nResults for: {prop_name}")
#         print(f" Test MSE:   {mean_mse:.5f} +/- {std_mse:.5f}")
#         print(f" Test MAE:   {mean_mae:.5f} +/- {std_mae:.5f}")
#         print(f" Test RMSE:  {mean_rmse:.5f} +/- {std_rmse:.5f}")
#         print(f" Test R2:    {mean_r2:.5f} +/- {std_r2:.5f}")

#     # Save results
#     os.makedirs(output_dir, exist_ok=True) 
#     scores_filename = f'{filename_prefix}_nested_cv_scores{filename_ending}.json'
#     scores_filepath = output_dir / scores_filename
#     with open(scores_filepath, 'w') as f:
#         json.dump(final_test_scores, f, indent=4)
#     print(f"\nFinal scores saved to '{scores_filepath}'")


#     predictions_filename = f'{filename_prefix}_nested_cv_predictions{filename_ending}.json'
#     predictions_filepath = output_dir / predictions_filename
#     with open(predictions_filepath, 'w') as f:
#         json.dump(final_fold_predictions, f, indent=4)
#     print(f"\nFinal predictions saved to '{predictions_filepath}'")


#     params_filename = f'{filename_prefix}_nested_cv_hyperparameters{filename_ending}.json'
#     params_filepath = output_dir / params_filename
#     with open(params_filepath, 'w') as f:
#         json.dump(final_fold_best_params, f, indent=4)
#     print(f"\nFinal hyperparameters saved to '{params_filepath}'")


def nested_cross_validation(filename_prefix,
                            prop_name,
                            model_naming_schema,
                            models_dir,
                            dataset_dir,
                            scalers,
                            model_class,
                            epochs,
                            hidden_dim_list,
                            dropout_range,
                            lr_range,
                            device,
                            n_trials,
                            outer_folds,
                            inner_folds,
                            batch_size,
                            output_dir,
                            filename_ending='',
                            patience = 10
                            ):
    """
    Performs importance analysis for node & edge features using a given model with
    nested cross-validation. Uses the objective_ElasticNet, an Optuna objective
    function for a single ElasticNet regularization hyperparameter tuning trial,
    to find the optimal hyperparameters.

    The ElasticNet regularization is run for n_trials inside the inner cv fold.

    Outputs a JSON file containing a dictionary where each key is a property name,
    and the value is another dictionary mapping each node & edge feature name to
    its averaged importance score.

    Output schema: {filename_prefix}_nested_cv_scores_{prop_name}{filename_ending}.json
                   {filename_prefix}_nested_cv_predictions_{prop_name}{filename_ending}.json
                   {filename_prefix}_nested_cv_hyperparameters_{prop_name}{filename_ending}.json

    Args:
        filename_prefix (str): The dataset name and output files prefix. e.g., 'qm9'.
        prop_names (List[str]): A list of the target property names to analyze. e.g., ['Dipole_moment', 'U', 'Cv'].
        model_naming_schema (str): The name for saving the models e.g., 'GINE_globals'.
        models_dir (str): Path to the directory to save the files.
        dataset_dir (str): Path to the directory containing the dataset files.
        scalers (dictionary): Dictionary with keys property names and values path to scalers
        model_class (torch.nn.Module): A model class to use for tuning.
        epochs (int): Number of maximum epochs to run training for that fold.
        patience (int): Number of epochs to wait for improvement before stopping training.
        hidden_dim_list  (List[int]): List of hidden dimensions to try. e.g., [128, 256].
        dropout_range (List[int]): List of the minimum and maximum value to test for the `dropout_rate` hyperparameter.
        lr_range (List[float]): List of the minimum and maximum value to test for the `learning_rate` hyperparameter.
        device (torch.device): cuda or cpu.
        n_trials (int): The number of Optuna trials to run for each inner loop.
        outer_folds (int): The number of splits for the outer cross-validation loop.
        inner_folds (int): The number of splits for the inner cross-validation loop.
        batch_size (int): Number of molecules per training batch. More is faster but requires more vram.
        output_dir (str): Path to the directory to save the JSON file.
        filename_ending (str, optional): The output files suffix.
        l1_ratio (float, optional): The L1 ratio for Elastic Net. Defaults to 1.0 (Lasso).
    """

    # This will store the final test scores from each outer fold
    final_test_scores = {prop_name: []}
    final_fold_predictions = {prop_name: []}
    final_fold_best_params = {prop_name: []}

    print(f"\n======= Nested CV for: {prop_name} =======")

    # Data Loading
    dataset_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
    dataset_path = dataset_dir / dataset_filename
    try:
        full_dataset = torch.load(dataset_path, weights_only=False)
        print(f"Loaded {prop_name} dataset")
    except FileNotFoundError:
            print(f"ERROR: Dataset not found at {dataset_path}")
            print("Ensure the dataset pruning cell has been run first.")

    # Determine feature sizes from the first graph
    num_node_features = full_dataset[0].x.shape[1]
    num_edge_features = full_dataset[0].edge_attr.shape[1]
    num_global_features = full_dataset[0].u.shape[1]

    # Implement nested cross validation
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    # ------- Start nested cross-validation -------
    print(" Starting nested cross-validation...")

    for fold_idx, (outer_train_index, outer_test_index) in enumerate(outer_cv.split(full_dataset)):
        print(f'Fold {fold_idx+1} out of {outer_folds}')

        # Create DataLoaders
        outer_train_dataset = Subset(full_dataset, outer_train_index)
        outer_test_dataset = Subset(full_dataset, outer_test_index)
        outer_test_loader = DataLoader(outer_test_dataset, batch_size=batch_size, shuffle=False)

        # Optuna Hyperparameter Study on the outer train_val set
        print(" Running Optuna to find best hyperparameters for this fold...")

        # We define the objective function inside the loop to capture the correct dataset split
        def objective_inner_cv(trial):
            # Suggest Hyperparameters
            hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_list)
            dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
            lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)

            val_losses = []

            # Inner loop
            for inner_train_idx, inner_val_idx in inner_cv.split(outer_train_dataset):
                inner_train_dataset = Subset(outer_train_dataset, inner_train_idx)
                inner_val_dataset = Subset(outer_train_dataset, inner_val_idx)
                inner_train_loader = DataLoader(inner_train_dataset, batch_size=batch_size, shuffle=True)
                inner_val_loader = DataLoader(inner_val_dataset, batch_size=batch_size, shuffle=False)

                model = model_class(
                    node_features=num_node_features,
                    edge_features=num_edge_features,
                    global_features=num_global_features,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    dropout_rate=dropout_rate
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.MSELoss()

                # Train for a smaller, fixed number of epochs for speed
                for _ in range(60):
                    train(model, inner_train_loader, optimizer, loss_fn, device)

                val_loss = test(model, inner_val_loader, loss_fn, device)
                val_losses.append(val_loss)

            return np.mean(val_losses)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective_inner_cv, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")

        # Train the final model on the outer training set
        model_name = f"{model_naming_schema}_{prop_name}_fold_{fold_idx+1}"

        # Instantiate the final model for training
        final_fold_model = model_class(
            node_features=num_node_features,
            edge_features=num_edge_features,
            global_features=num_global_features,
            hidden_dim=best_params['hidden_dim'],
            output_dim=1,
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        outer_train_loader = DataLoader(outer_train_dataset, batch_size=batch_size, shuffle=True)

        trained_model, _ = train_with_early_stopping(
            final_fold_model,
            model_name,
            models_dir,
            filename_prefix,
            outer_train_loader,
            outer_test_loader,
            epochs=epochs,
            lr=best_params['lr'],
            device=device,
            patience=patience,
            filename_ending=filename_ending)

        # Evaluate on the outer test set
        print(" Evaluating on outer test fold...")
        trained_model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for data in outer_test_loader:
                data = data.to(device)
                all_preds.append(trained_model(data).cpu().numpy())
                all_true.append(data.y.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        true = np.concatenate(all_true, axis=0)

        scaler = scalers[prop_name]
        unscaled_preds = scaler.inverse_transform(preds)
        unscaled_true = scaler.inverse_transform(true)

        mse = mean_squared_error(unscaled_true, unscaled_preds)
        mae = mean_absolute_error(unscaled_true, unscaled_preds)
        rmse = root_mean_squared_error(unscaled_true, unscaled_preds)
        r2 = r2_score(unscaled_true, unscaled_preds)

        print(f"\n Fold Test MSE: {mse:.5f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, R2: {r2:.5f}")

        scores_data = {
            'fold_index': fold_idx,
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2)
        }
        final_test_scores[prop_name].append(scores_data)

        fold_prediction_data = {
            'fold_index': fold_idx,
            'predictions': unscaled_preds.tolist(),
            'true_values': unscaled_true.tolist()
        }

        final_fold_predictions[prop_name].append(fold_prediction_data)

        params_data = best_params.copy() # Make a copy
        params_data['fold_index'] = fold_idx
        final_fold_best_params[prop_name].append(params_data)

    # Report results
    print(f"\n\n======= Nested cross validation results for {prop_name} (Mean +/- Std dev across folds) =======")
    for prop_name, scores_list in final_test_scores.items():
        mse_scores = [s['MSE'] for s in scores_list]
        mae_scores = [s['MAE'] for s in scores_list]
        rmse_scores = [s['RMSE'] for s in scores_list]
        r2_scores = [s['R2'] for s in scores_list]

        # Calculate mean and standard deviation
        mean_mse, std_mse = np.mean(mse_scores), np.std(mse_scores)
        mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
        mean_rmse, std_rmse = np.mean(rmse_scores), np.std(rmse_scores)
        mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)

        print(f" Test MSE:   {mean_mse:.5f} +/- {std_mse:.5f}")
        print(f" Test MAE:   {mean_mae:.5f} +/- {std_mae:.5f}")
        print(f" Test RMSE:  {mean_rmse:.5f} +/- {std_rmse:.5f}")
        print(f" Test R2:    {mean_r2:.5f} +/- {std_r2:.5f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True) 
    scores_filename = f'{filename_prefix}_nested_cv_scores_{prop_name}{filename_ending}.json'
    scores_filepath = output_dir / scores_filename
    with open(scores_filepath, 'w') as f:
        json.dump(final_test_scores, f, indent=4)
    print(f"\nFinal scores saved to '{scores_filepath}'")


    predictions_filename = f'{filename_prefix}_nested_cv_predictions_{prop_name}{filename_ending}.json'
    predictions_filepath = output_dir / predictions_filename
    with open(predictions_filepath, 'w') as f:
        json.dump(final_fold_predictions, f, indent=4)
    print(f"\nFinal predictions saved to '{predictions_filepath}'")


    params_filename = f'{filename_prefix}_nested_cv_hyperparameters_{prop_name}{filename_ending}.json'
    params_filepath = output_dir / params_filename
    with open(params_filepath, 'w') as f:
        json.dump(final_fold_best_params, f, indent=4)
    print(f"\nFinal hyperparameters saved to '{params_filepath}'")



# ===== Train loop =====


# def train_loop(filename_prefix,
#                prop_names,
#                index_dir,
#                hyper_dir,
#                model_naming_schema,
#                models_dir,
#                dataset_dir,
#                scalers,
#                model_class,
#                epochs,
#                device,
#                batch_size,
#                filename_ending='',
#                patience = 10
#                ):
    
#     histories = {}
#     trained_models = {}

#     # Load indexes from CSV files
#     print("Loading train, validation, test indexes from CSV...")
#     os.makedirs(index_dir, exist_ok=True) 
#     train_path = index_dir / f'{filename_prefix}_index_train{filename_ending}.csv'
#     val_path = index_dir / f'{filename_prefix}_index_validation{filename_ending}.csv'
#     test_path = index_dir / f'{filename_prefix}_index_test{filename_ending}.csv'

#     try:
#         index_train_df = pd.read_csv(train_path)
#         index_val_df = pd.read_csv(val_path)
#         index_test_df = pd.read_csv(test_path)

#         # Convert the DataFrame columns to NumPy arrays (or lists) for indexing
#         train_index = index_train_df['index'].values
#         val_index = index_val_df['index'].values
#         test_index = index_test_df['index'].values

#         # Concatenate (combine) train and validation sets to use for training.
#         train_concat_index = np.concatenate([train_index, val_index])

#         print(f"Train samples: {len(train_concat_index)}, Test samples: {len(test_index)}")
#     except FileNotFoundError as e:
#         print(f"\nERROR: Could not find one or more index files.")
#         print(f"Specifically, the file not found was: {e.filename}")
#         print("Ensure the data splitting cell has been run first.")

#     # Uncomment if Hyperparameter tuning is not run
#     os.makedirs(hyper_dir, exist_ok=True) 
#     params_filename = f'{filename_prefix}_best_hyperparameters{filename_ending}.json'
#     params_filepath = hyper_dir / params_filename
#     try:
#         with open(params_filepath, 'r') as f:
#             best_params = json.load(f)
#             print(f"Successfully loaded best parameters from {params_filepath}")
#     except FileNotFoundError:
#         print(f"ERROR: Feature name file not found at {params_filepath}")
#         print("Ensure the hyperparameter tuning cell has been run first.")


#     # ------- Main Loop -------
#     for prop_name in prop_names:
#         print(f"\n======= Training for: {prop_name} =======")

#         # Data Loading
#         dataset_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
#         dataset_path = dataset_dir / dataset_filename
#         try:
#             full_dataset = torch.load(dataset_path, weights_only=False)
#             print(f"Loaded {prop_name} dataset")
#         except FileNotFoundError:
#                 print(f"ERROR: Dataset not found at {dataset_path}")
#                 print("Ensure the dataset pruning cell has been run first.")

#         # Determine feature sizes from the first graph
#         num_node_features = full_dataset[0].x.shape[1]
#         num_edge_features = full_dataset[0].edge_attr.shape[1]
#         num_global_features = full_dataset[0].u.shape[1]

#         # Split the train and test sets
#         train_dataset = Subset(full_dataset, train_concat_index)
#         test_dataset = Subset(full_dataset, test_index)

#         # Create DataLoaders
#         # We will use both the train and validation datasets for training.
#         # Validation will happen on the completely unseen test subset.
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#         print("Subsets and DataLoaders created successfully.")

#         # ------- Train the GINE model -------
#         # comment this to skip retraining
#         model_name = f"{model_naming_schema}_{prop_name}"
#         current_params = best_params[prop_name]

#         model = model_class(
#             node_features=num_node_features,
#             edge_features=num_edge_features,
#             global_features=num_global_features,
#             hidden_dim=current_params['hidden_dim'],
#             output_dim=1,
#             dropout_rate=current_params['dropout_rate']
#         ).to(device)

        
#         # Run for patience = 1 for faster results
#         trained_model, model_history = train_with_early_stopping(
#             model,
#             model_name,
#             models_dir,
#             filename_prefix,
#             train_loader,
#             test_loader,
#             epochs=epochs,
#             lr=current_params['lr'],
#             device=device,
#             patience=patience,
#             filename_ending=filename_ending
#         )

#         #------- End of training -------
#         # Store history and models for later
#         histories[prop_name] = model_history
#         trained_models[prop_name] = trained_model

#         # Results
#         evaluate_and_plot_one_property(trained_model, model_name, test_loader, device, scalers, prop_name)
#         plot_training_history_one_property(model_history, model_name, prop_name)
#         calculate_mae_one_property(trained_model, model_name, test_loader, device, scalers, prop_name)

#         print(f"\n======= Finished processing for {prop_name} =======\n")


def train_loop(filename_prefix,
               prop_name,
               index_dir,
               hyper_dir,
               model_naming_schema,
               models_dir,
               dataset_dir,
               scalers,
               model_class,
               epochs,
               device,
               batch_size,
               filename_ending='',
               patience = 10
               ):
    
    histories = {}
    trained_models = {}

    # Load indexes from CSV files
    print("Loading train, validation, test indexes from CSV...")
    os.makedirs(index_dir, exist_ok=True) 
    train_path = index_dir / f'{filename_prefix}_index_train{filename_ending}.csv'
    val_path = index_dir / f'{filename_prefix}_index_validation{filename_ending}.csv'
    test_path = index_dir / f'{filename_prefix}_index_test{filename_ending}.csv'

    try:
        index_train_df = pd.read_csv(train_path)
        index_val_df = pd.read_csv(val_path)
        index_test_df = pd.read_csv(test_path)

        # Convert the DataFrame columns to NumPy arrays (or lists) for indexing
        train_index = index_train_df['index'].values
        val_index = index_val_df['index'].values
        test_index = index_test_df['index'].values

        # Concatenate (combine) train and validation sets to use for training.
        train_concat_index = np.concatenate([train_index, val_index])

        print(f"Train samples: {len(train_concat_index)}, Test samples: {len(test_index)}")
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find one or more index files.")
        print(f"Specifically, the file not found was: {e.filename}")
        print("Ensure the data splitting cell has been run first.")


    # ------- Main Loop -------
    print(f"\n======= Training for: {prop_name} =======")

    # Load best hyperparameters
    os.makedirs(hyper_dir, exist_ok=True) 
    params_filename = f'{filename_prefix}_best_hyperparameters_{prop_name}{filename_ending}.json'
    params_filepath = hyper_dir / params_filename
    try:
        with open(params_filepath, 'r') as f:
            best_params = json.load(f)
            current_params = best_params[prop_name]
            print(f"Successfully loaded best parameters from {params_filepath}")
    except FileNotFoundError:
        print(f"ERROR: Feature name file not found at {params_filepath}")
        print("Ensure the hyperparameter tuning cell has been run first.")

    # Data Loading
    dataset_filename = f'{filename_prefix}_pruned_dataset_{prop_name}{filename_ending}.pt'
    dataset_path = dataset_dir / dataset_filename
    try:
        full_dataset = torch.load(dataset_path, weights_only=False)
        print(f"Loaded {prop_name} dataset")
    except FileNotFoundError:
            print(f"ERROR: Dataset not found at {dataset_path}")
            print("Ensure the dataset pruning cell has been run first.")

    # Determine feature sizes from the first graph
    num_node_features = full_dataset[0].x.shape[1]
    num_edge_features = full_dataset[0].edge_attr.shape[1]
    num_global_features = full_dataset[0].u.shape[1]

    # Split the train and test sets
    train_dataset = Subset(full_dataset, train_concat_index)
    test_dataset = Subset(full_dataset, test_index)

    # Create DataLoaders
    # We will use both the train and validation datasets for training.
    # Validation will happen on the completely unseen test subset.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Subsets and DataLoaders created successfully.")

    # ------- Train the GINE model -------
    # comment this to skip retraining
    model_name = f"{model_naming_schema}_{prop_name}"

    model = model_class(
        node_features=num_node_features,
        edge_features=num_edge_features,
        global_features=num_global_features,
        hidden_dim=current_params['hidden_dim'],
        output_dim=1,
        dropout_rate=current_params['dropout_rate']
    ).to(device)

        
    # Run for patience = 1 for faster results
    trained_model, model_history = train_with_early_stopping(
        model,
        model_name,
        models_dir,
        filename_prefix,
        train_loader,
        test_loader,
        epochs=epochs,
        lr=current_params['lr'],
        device=device,
        patience=patience,
        filename_ending=filename_ending
    )

    #------- End of training -------

    # Results
    evaluate_and_plot_one_property(trained_model, model_name, test_loader, device, scalers, prop_name)
    plot_training_history_one_property(model_history, model_name, prop_name)
    calculate_mae_one_property(trained_model, model_name, test_loader, device, scalers, prop_name)

    print(f"\n======= Finished processing for {prop_name} =======\n")

    return trained_model, model_history