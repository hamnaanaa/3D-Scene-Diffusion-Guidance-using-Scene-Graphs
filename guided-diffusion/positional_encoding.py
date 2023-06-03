import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.create_positional_encoding(d_model, max_len)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        return x

    def create_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        positional_encoding = torch.zeros(1, max_len, d_model)
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return positional_encoding
