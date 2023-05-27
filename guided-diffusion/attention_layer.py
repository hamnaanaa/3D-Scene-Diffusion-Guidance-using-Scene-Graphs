import torch
import torch.nn as nn


class ModifiedMultiheadAttention(nn.Module):
    """
    Multi-Head Attention Layer with Residual Connections and Layer Normalization followed by a linear projection.

    Args:
        input_dim (int): Dimension of the input matrix.
        target_dim (int): Dimension of the target space.
        num_heads (int): Number of attention heads, must be a divisor of input_dim.

    """
    def __init__(self, input_dim, target_dim, num_heads):
        super(ModifiedMultiheadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Number of heads must be a divisor of input_dim."

        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.output_projection = nn.Linear(input_dim, target_dim)

    def forward(self, inputs):
        B, N, _ = inputs.size()

        # Compute multi-head attention
        attended_values, _ = self.attention(inputs, inputs, inputs)

        # Apply layer normalization and residual connection
        normalized_values = self.layer_norm(inputs + attended_values)

        # Project the normalized_values to the output dimension
        output = self.output_projection(normalized_values)

        return output
