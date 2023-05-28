import math
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    """
    TimeEmbedding module for generating time-based embeddings.

    Args:
        dim (int): The output dimension of the time embeddings. It should be an even number.

    Example:
        >>> time_embedding = TimeEmbedding(dim=128)
        >>> inputs = torch.tensor([0, 1, 2, 3, 4])  # Assuming inputs represent time steps
        >>> embeddings = time_embedding(inputs)
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Generate time-based embeddings.

        Args:
            x (torch.Tensor): Input tensor representing time steps. It should be of shape (B,).

        Returns:
            torch.Tensor: Time embeddings of shape (B, dim).

        Note:
            - The input tensor `x` needs to be of type torch.Tensor with length B.
            - The output tensor will have shape (B, dim) where dim is the specified output dimension.
            - The specified output dimension `dim` should be an even number.

        Example:
            >>> time_embedding = TimeEmbedding(dim=128)
            >>> inputs = torch.tensor([0, 1, 2, 3, 4])  # Assuming inputs represent time steps
            >>> embeddings = time_embedding(inputs)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
