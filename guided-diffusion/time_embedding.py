import math
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    """
    Input: t [B]
    Output: t_embedd [B, F]
    This module is used to embedd the time t into a vector of size F.
    The embedding is done by using the sine and cosine functions.
    """
    
    def __init__(self, dim): # set output dim F as an EVEN number!
        super().__init__()
        self.dim = dim

    def forward(self, x): # x needs to be of type torch.tensor with length N!
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
