from torch import nn
from torch_geometric.nn import RGCNConv

class RelationalRGCN(nn.Module):
    """
    Module using the Relational Graph Convolutional Network (RGCN) architecture.

    Args:
        in_channels (int): Number of input channels/features.
        h_channels_list (list[int]): List of hidden channel sizes for each layer, set to [] for no hidden layers.
        out_channels (int): Number of output channels/features.
        num_relations (int): Number of edge relation types in the graph.
        num_bases (int, optional): Number of bases for relation weights (default: None).
        aggr (str, optional): Aggregation method for message passing ('mean', 'add', 'max', 'min') (default: 'mean').
        dp_rate (float, optional): Dropout rate (default: 0.1).
        bias (bool, optional): If set to False, the layer will not learn an additive bias (default: True).

    Attributes:
        num_layers (int): Total number of layers in the RGCN.
        layers (torch.nn.ModuleList): List of RGCNConv, ReLU, and Dropout layers.

    """
    def __init__(
        self, 
        in_channels,
        h_channels_dims,
        out_channels, 
        num_relations,
        num_bases=None,
        aggr='mean',
        activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        dp_rate=0.1, 
        bias=True
    ):
        super(RelationalRGCN, self).__init__()
        self.num_hidden_layers = len(h_channels_dims)
        self.layers = []
        
        for i in range(self.num_hidden_layers):
            source_channels = in_channels if i == 0 else h_channels_dims[i - 1]
            target_channels = h_channels_dims[i]
            self.layers += [
                RGCNConv(
                    in_channels=source_channels,
                    out_channels=target_channels,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    aggr=aggr,
                    bias=bias
                ),
                activation,
                nn.Dropout(p=dp_rate)
            ]
        self.layers += [
            RGCNConv(
                in_channels=h_channels_dims[-1] if self.num_hidden_layers > 0 else in_channels,
                out_channels=out_channels,
                num_relations=num_relations,
                num_bases=num_bases,
                aggr=aggr,
                bias=bias
            ),
            activation
        ]
        
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass of the RelationalRGCN module.

        Args:
            x (torch.Tensor): Input features of shape (B, N, C) = (batch_size, num_nodes, in_channels).
            edge_index (torch.Tensor): Graph edge indices of shape (B, 2, E) = (batch_size, 2, num_edges).
            edge_type (torch.Tensor): Edge type indices of shape (B, E) = (batch_size, num_edges).

        Returns:
            torch.Tensor: Output features of shape (B, N, H1) = (batch_size, num_nodes, out_channels).

        """
        for layer in self.layers:
            if isinstance(layer, RGCNConv):
                x = layer(x, edge_index, edge_type)
            else:
                x = layer(x)
        return x