from torch import nn
import torch_geometric.transforms as T
from torch_geometric.nn import RGCNConv

class EncoderRGCN(nn.Module):
    def __init__(
        self, 
        in_channels,
        h_channels_list,
        out_channels, 
        num_relations,
        num_bases=None, 
        aggr='mean',
        dp_rate=0.1, 
        bias=True
    ):
        super(EncoderRGCN, self).__init__()
        self.num_layers = len(h_channels_list) + 1
        self.layers = []
        
        for i in range(self.num_layers - 1):
            in_channels = in_channels if i == 0 else h_channels_list[i - 1]
            out_channels = h_channels_list[i]
            self.layers += [
                RGCNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    aggr=aggr,
                    bias=bias
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dp_rate)
            ]
        self.layers += [
            RGCNConv(
                in_channels=h_channels_list[-1],
                out_channels=out_channels,
                num_relations=num_relations,
                num_bases=num_bases,
                aggr=aggr,
                bias=bias
            ),
            nn.ReLU(inplace=True) # TODO: other final activation function here?
        ]
        
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x, edge_index, edge_type):
        for layer in self.layers:
            if isinstance(layer, RGCNConv):
                x = layer(x, edge_index, edge_type)
            else:
                x = layer(x)
        return x