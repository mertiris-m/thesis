from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

# # This is the exact copy of the source code for torch_geometric.nn.conv.gin_conv
# # The logic behid this is to be able to access the inner linear layer for feature selection


# class GINEConv_Custom(MessagePassing):
#     r"""The modified :class:`GINConv` operator from the `"Strategies for
#     Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
#     paper.

#     .. math::
#         \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
#         \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{PReLU}
#         ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

#     that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
#     the aggregation procedure.

#     Args:
#         nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
#             maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
#             shape :obj:`[-1, out_channels]`, *e.g.*, defined by
#             :class:`torch.nn.Sequential`.
#         eps (float, optional): (Initial) :math:`\epsilon`-value.
#             (default: :obj:`0.`)
#         train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
#             will be a trainable parameter. (default: :obj:`False`)
#         edge_dim (int, optional): Edge feature dimensionality. If set to
#             :obj:`None`, node and edge feature dimensionality is expected to
#             match. Other-wise, edge features are linearly transformed to match
#             node feature dimensionality. (default: :obj:`None`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.

#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`,
#           edge features :math:`(|\mathcal{E}|, D)` *(optional)*
#         - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
#           :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
#     """
#     def __init__(self, nn: torch.nn.Module, eps: float = 0.,
#                  train_eps: bool = False, edge_dim: Optional[int] = None,
#                  **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self.nn = nn
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.empty(1))
#         else:
#             self.register_buffer('eps', torch.empty(1))
#         if edge_dim is not None:
#             if isinstance(self.nn, torch.nn.Sequential):
#                 nn = self.nn[0]
#             if hasattr(nn, 'in_features'):
#                 in_channels = nn.in_features
#             elif hasattr(nn, 'in_channels'):
#                 in_channels = nn.in_channels
#             else:
#                 raise ValueError("Could not infer input channels from `nn`.")
#             self.lin = Linear(edge_dim, in_channels)

#         else:
#             self.lin = None
        
#         # self.prelu = nn.PReLU() # ---------------- this is added ----------------
#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.nn)
#         self.eps.data.fill_(self.initial_eps)
#         if self.lin is not None:
#             self.lin.reset_parameters()


#     def forward(
#         self,
#         x: Union[Tensor, OptPairTensor],
#         edge_index: Adj,
#         edge_attr: OptTensor = None,
#         size: Size = None,
#     ) -> Tensor:

#         if isinstance(x, Tensor):
#             x = (x, x)

#         # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

#         x_r = x[1]
#         if x_r is not None:
#             out = out + (1 + self.eps) * x_r

#         return self.nn(out)


#     def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
#         if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
#             raise ValueError("Node and edge feature dimensionalities do not "
#                              "match. Consider setting the 'edge_dim' "
#                              "attribute of 'GINEConv'")

#         if self.lin is not None:
#             edge_attr = self.lin(edge_attr)

#         return (x_j + edge_attr).relu # ---------------- this is changed ----------------

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(nn={self.nn})'



# class GINE_Custom(nn.Module):
#     """
#     A GINEConv model. The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
#     Added seperate simple MLPs for the required NN before the GINE layer.
#     Two seperate GINE layers are used and then aggregated.
#     Added a dropout method near the start and near the end, for the node/edge features and general regularizaion.
#     Each Gine layer is passed through a ReLU activation layer for non-linearity.
#     A final larger non linear block (MLP) is used to reduce dimensionality.

    
#     Args:
#         node_features (int): The dimensionality of the input node features.
#         edge_features (int): The dimensionality of the input edge features.
#         hidden_dim (int): The dimensionality of the hidden layers.
#         output_dim (int): The number of output properties to predict.
#         dropout_rate (float): The dropout probability.
#     """
#     def __init__(self, node_features: int, edge_features: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
#         super().__init__()
        
#         # First Layer
#         # GINEConv requires a neural network to process the node features.
#         # A simple MLP is used.
#         # However the simple ReLu layer discards every negative input - information is lost.
#         # So in order not to lose any information for the feature selection step that only
#         # uses the first GINEConv layer we decide to use the Parametric ReLU (PReLU) layer,
#         # which has a learnable negative slope for each negative value.
#         # For the same reasons the relu on the GINEConv_Custom message passing layer  
#         # is swapped with a prelu layer.
#         mlp1 = nn.Sequential(
#             nn.Linear(node_features, hidden_dim),
#             nn.PReLU(),
#             nn.Linear(hidden_dim, hidden_dim), 
#         )

#         # First GINEConv layer. It takes a linear layer for edge features and the MLP for node features.
#         self.conv1 = GINEConv_Custom(nn=mlp1, edge_dim=edge_features, train_eps=True)

#         # Second GINEConv Layer
#         mlp2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.conv2 = GINEConv_Custom(nn=mlp2, edge_dim=edge_features, train_eps=True)
        
#         # The pooling layer aggregates node embeddings to produce a single graph-level embedding.
#         # global_add_pool is often a strong choice for molecular graphs.
#         self.pool = global_add_pool
        
#         # Dropout rate for regularization, also used for Monte Carlo dropout at inference.
#         self.dropout_rate = dropout_rate
        
#         # Final MLP
#         # This block maps the final graph embedding to the desired number of output properties.
#         self.fc_block = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_rate),
#             nn.Linear(hidden_dim * 2, output_dim)
#         )

#     def forward(self, data: 'torch_geometric.data.Batch') -> torch.Tensor:
#         """
#         Performs the forward pass of the model.
        
#         Args:
#             data (torch_geometric.data.Batch): A batch of graph data.
            
#         Returns:
#             torch.Tensor: The model's predictions.
#         """
#         # Unpack the data object
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
#         # 1. First GINE layer, followed by activation and dropout
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         # F.dropout is used to ensure dropout is active during model.train() for MC-Dropout
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
#         # 2. Second GINE layer, followed by activation
#         x = F.relu(self.conv2(x, edge_index, edge_attr))
        
#         # 3. Global pooling to get a graph-level representation
#         x_pooled = self.pool(x, batch)
        
#         # 4. Final MLP block to produce the output
#         out = self.fc_block(x_pooled)
        
#         return out


class GINE_Custom(nn.Module):
    """
    A GINEConv model. The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    Added seperate simple MLPs for the required NN before the GINE layer.
    Two seperate GINE layers are used and then aggregated.
    Added a dropout method near the start and near the end.
    Each Gine layer is passed through a ReLU activation layer for non-linearity.
    A final larger MLP is used to reduce dimensionality.

    
    Args:
        node_features (int): The dimensionality of the input node features.
        edge_features (int): The dimensionality of the input edge features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The number of output properties to predict.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # The projector layers.
        # They project node and edge features to the hidden dimension
        self.node_projector = nn.Linear(node_features, hidden_dim)
        self.edge_projector = nn.Linear(edge_features, hidden_dim)

        # First Layer
        # GINEConv requires a neural network to process the node features.
        # A simple MLP is used.
        # However the simple ReLu layer discards every negative input - information is lost.
        # So in order not to lose any information for the feature selection step that only
        # uses the first GINEConv layer we decide to use the Parametric ReLU (PReLU) layer,
        # which has a learnable negative slope for each negative value.

        mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
        )

        # First GINEConv layer. It takes a linear layer for edge features and the MLP for node features.
        self.conv1 = GINEConv(nn=mlp1, train_eps=True)

        # After the GINEConv layer a PReLU layer is used
        self.prelu = nn.PReLU()

        # Second GINEConv Layer
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn=mlp2, train_eps=True)
        
        # The pooling layer aggregates node embeddings to produce a single graph-level embedding.
        # global_add_pool is often a strong choice for molecular graphs.
        self.pool = global_add_pool
        
        # Dropout rate for regularization, also used for Monte Carlo dropout at inference.
        self.dropout_rate = dropout_rate
        
        # Final MLP
        # This block maps the final graph embedding to the desired number of output properties.
        # Same as the mlp for global features to ensure comparability.
        # Since we add the global features before this step a larger, more complex MLP is used
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
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

        # Project the node and edge features to the hidden dimension
        x = self.node_projector(x)
        edge_attr = self.edge_projector(edge_attr)
        
        # 1. First GINE layer, followed by PReLU activation and dropout
        x = self.prelu(self.conv1(x, edge_index, edge_attr))
        # F.dropout is used to ensure dropout is active during model.train() for MC-Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 2. Second GINE layer, followed by ReLU activation
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # 3. Global pooling to get a graph-level representation
        x_pooled = self.pool(x, batch)
        
        # 4. Final MLP block to produce the output
        out = self.fc_block(x_pooled)
        
        return out



class GINE_Custom_With_Globals(nn.Module):
    """
    A GINEConv model. The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    Modified to support global features from a data.u object.
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
    def __init__(self, node_features: int, edge_features: int, global_features: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # The projector layers.
        # They project node and edge features to the hidden dimension
        self.node_projector = nn.Linear(node_features, hidden_dim)
        self.edge_projector = nn.Linear(edge_features, hidden_dim)

        # First Layer
        # GINEConv requires a neural network to process the node features.
        # A simple MLP is used.
        # However the simple ReLu layer discards every negative input - information is lost.
        # So in order not to lose any information for the feature selection step that only
        # uses the first GINEConv layer we decide to use the Parametric ReLU (PReLU) layer,
        # which has a learnable negative slope for each negative value.
        mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
        )

        # After the GINEConv layer a PReLU layer is used
        self.prelu = nn.PReLU()

        # First GINEConv layer. It takes a linear layer for edge features and the MLP for node features.
        self.conv1 = GINEConv(nn=mlp1, train_eps=True)

        # Second GINEConv Layer
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn=mlp2, train_eps=True)
        
        # The pooling layer aggregates node embeddings to produce a single graph-level embedding.
        # global_add_pool is often a strong choice for molecular graphs.
        self.pool = global_add_pool
        
        # Dropout rate for regularization, also used for Monte Carlo dropout at inference.
        self.dropout_rate = dropout_rate
        
        # Final MLP
        # This block maps the final graph embedding to the desired number of output properties.
        # Since we add the global features before this step a larger, more complex MLP is used
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim + global_features, hidden_dim * 2),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
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
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u

        # Project the node and edge features to the hidden dimension
        x = self.node_projector(x)
        edge_attr = self.edge_projector(edge_attr)
        
        # 1. First GINE layer, followed by PReLU activation and dropout
        x = self.prelu(self.conv1(x, edge_index, edge_attr))
        # F.dropout is used to ensure dropout is active during model.train() for MC-Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 2. Second GINE layer, followed by ReLU activation
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # 3. Global pooling to get a graph-level representation
        x_pooled = self.pool(x, batch)

        # 4. Imcorporate global features
        # u has shape: [batch_size, num_global_features]
        combined = torch.cat([x_pooled, u], dim=1) # Shape: [batch_size, hidden_dim + num_global_features]
        
        # 5. Final MLP block to produce the output
        out = self.fc_block(combined)
        
        return out