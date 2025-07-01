import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_add_pool


""" With GINEConv layers https://doi.org/10.48550/arXiv.1905.12265"""
    

class GINE(nn.Module):
    """
    A GINEConv model. The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    Added seperate simple MLPs for the required NN before the GINE layer.
    Two seperate GINE layers are used and then aggregated.
    Added a dropout method near the start and near the end, for the node/edge features and general regularizaion.
    Each Gine layer is passed through a ReLU activation layer for non-linearity.
    A final larger non linear block (MLP) is used to reduce dimensionality.

    
    Args:
        node_features (int): The dimensionality of the input node features.
        edge_features (int): The dimensionality of the input edge features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The number of output properties to predict.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # First Layer
        # GINEConv requires a neural network to process the node features.
        # A simple MLP is used
        mlp1 = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
        )

        # First GINEConv layer. It takes a linear layer for edge features and the MLP for node features.
        self.conv1 = GINEConv(nn=mlp1, edge_dim=edge_features, train_eps=True)

        # Second GINEConv Layer
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn=mlp2, edge_dim=edge_features, train_eps=True)
        
        # The pooling layer aggregates node embeddings to produce a single graph-level embedding.
        # global_add_pool is often a strong choice for molecular graphs.
        self.pool = global_add_pool
        
        # Dropout rate for regularization, also used for Monte Carlo dropout at inference.
        self.dropout_rate = dropout_rate
        
        # Final MLP
        # This block maps the final graph embedding to the desired number of output properties.
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, data: 'torch_geometric.data.Batch') -> torch.Tensor:
        """
        Performs the forward pass of the model.
        
        Args:
            data (torch_geometric.data.Batch): A batch of graph data.
            
        Returns:
            torch.Tensor: The model's predictions.
        """
        # Unpack the data object
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. First GINE layer, followed by activation and dropout
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # F.dropout is used to ensure dropout is active during model.train() for MC-Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 2. Second GINE layer, followed by activation
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # 3. Global pooling to get a graph-level representation
        x_pooled = self.pool(x, batch)
        
        # 4. Final MLP block to produce the output
        out = self.fc_block(x_pooled)
        
        return out
    

# --- Example from ethane results ---

num_node_features = 14
num_edge_features = 6
num_properties = 3
hidden_dimension = 128 # DimeNet default
dropout = 0.2

# Instantiate the model
model = GINE(
    node_features=num_node_features,
    edge_features=num_edge_features,
    hidden_dim=hidden_dimension,
    output_dim=num_properties,
    dropout_rate=dropout
)

print("----- GINE Model Architecture -----")
print(model)
print("\nModel created successfully and ready to be used.")
