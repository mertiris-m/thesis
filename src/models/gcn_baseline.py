import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class GCN_baseline(nn.Module):
    """
    A simple GCN (Graph Convolutional Network) model for baseline  performance.
    (Does not use edge features) Same pipeline as the GINE model.
    
    Args:
        node_features (int): The dimensionality of the input node features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The number of output properties to predict.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, node_features: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # GCN Convolutional Layers
        # GCNConv only uses node features and the adjacency matrix.
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Pooling and dropout
        self.pool = global_add_pool
        self.dropout_rate = dropout_rate
        
        # Final MLP
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, data: 'torch_geometric.data.Batch') -> torch.Tensor:
        """
        Performs the forward pass of the model.
        
        NOTE: It ignores data.edge_attr.
        """
        # Unpack the data object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. First GCN layer, followed by activation and dropout
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 2. Second GCN layer, followed by activation
        x = F.relu(self.conv2(x, edge_index))
        
        # 3. Global pooling
        x_pooled = self.pool(x, batch)
        
        # 4. Final MLP block
        out = self.fc_block(x_pooled)
        
        return out

# --- Example from ethane results ---


num_node_features = 14
num_properties = 3
hidden_dimension = 128 # DimeNet default
dropout = 0.2

# Instantiate the baseline model
model = GCN_baseline(
    node_features=num_node_features,
    hidden_dim=hidden_dimension,
    output_dim=num_properties,
    dropout_rate=dropout
)

print("----- Baseline GCN Model Architecture -----")
print(model)
print("\nModel created successfully and ready to be used.")