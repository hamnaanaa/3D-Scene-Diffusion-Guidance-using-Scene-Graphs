import itertools
import numpy as np
import torch
import torch.nn as nn

from attention_layer import ModifiedMultiheadAttention
from relational_gcn import RelationalRGCN
from time_embedding import TimeEmbedding


class GuidedDiffusionNetwork(nn.Module):
    def __init__(
        self,
        # Attention block
        attention_in_dim,
        attention_out_dim,
        attention_num_heads,
        # Common RGCN parameters
        rgcn_num_relations,
        # Encoder RGCN block
        encoder_in_dim, 
        encoder_out_dim, 
        encoder_hidden_dims=f"{()}",
        encoder_num_bases=None,
        encoder_aggr='mean',
        encoder_activation="leakyrelu",
        encoder_dp_rate=0.1,
        encoder_bias=True,
        # Fusion block
        fusion_hidden_dims=f"{()}",
        fusion_num_bases=None,
        fusion_aggr='mean',
        fusion_activation="leakyrelu",
        fusion_dp_rate=0.1,
        fusion_bias=True,
        # Classifier-free guidance parameters
        cond_drop_prob=0.2,
    ):
        super(GuidedDiffusionNetwork, self).__init__()
        
        self.cond_drop_prob = cond_drop_prob
        
        # Instantiate the activation functions from the string
        if encoder_activation == "leakyrelu":
            encoder_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif encoder_activation == "relu":
            encoder_activation = nn.ReLU(inplace=True)
        elif encoder_activation == "silu":
            encoder_activation = nn.SiLU(inplace=True)
        elif encoder_activation == "tanh":
            encoder_activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation function {encoder_activation} is not implemented.")
        
        if fusion_activation == "leakyrelu":
            fusion_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif fusion_activation == "relu":
            fusion_activation = nn.ReLU(inplace=True)
        elif fusion_activation == "silu":
            fusion_activation = nn.SiLU(inplace=True)
        elif fusion_activation == "tanh":
            fusion_activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation function {fusion_activation} is not implemented.")
        
        # Instantiate hidden_dims tuples from the string
        encoder_hidden_dims, fusion_hidden_dims = eval(encoder_hidden_dims), eval(fusion_hidden_dims)
        
        self.attention_module = ModifiedMultiheadAttention(
            input_dim=attention_in_dim, 
            embed_dim=attention_out_dim, 
            num_heads=attention_num_heads # TODO: hyperparam vs. hardcode?
        )
        
        self.encoder_module = RelationalRGCN(
            in_channels=encoder_in_dim, 
            h_channels_dims=encoder_hidden_dims,
            out_channels=encoder_out_dim,
            num_relations=rgcn_num_relations, 
            num_bases=encoder_num_bases, 
            aggr=encoder_aggr,
            activation=encoder_activation,
            dp_rate=encoder_dp_rate, 
            bias=encoder_bias
        )
        
        self.time_embedding_module = TimeEmbedding(dim=attention_out_dim)
        
        self.fused_rgcn_module = RelationalRGCN(
            in_channels=attention_out_dim + encoder_out_dim,
            h_channels_dims=fusion_hidden_dims,
            out_channels=attention_in_dim,
            num_relations=rgcn_num_relations,
            num_bases=fusion_num_bases,
            aggr=fusion_aggr,
            activation=fusion_activation,
            dp_rate=fusion_dp_rate,
            bias=fusion_bias
        )
    
    # This forward method should return the output prediction of noise of the final relational GCN in shape [B, N, D]
    def forward(self, x, t, obj_cond, edge_cond, relation_cond, cond_drop_prob=None):
        """
        Forward pass of the GuidedDiffusionNetwork.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D] representing the initial input.
            t (torch.Tensor): Time tensor of shape [B] representing the corresponding timesteps.
            obj_cond (torch.Tensor): Object condition tensor of shape [B*N, C] representing the object condition.
            edge_cond (torch.Tensor): Edge condition tensor of shape [2, E] representing the edge condition.
            relation_cond (torch.Tensor): Relation condition tensor of shape [E] representing the corresponding relation type condition.
            cond_drop_prob (float, optional): Probability of dropping the classifier-free guidance. If none is provided, the default model's probability is used.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, D] representing the predicted noise of the final fused relational GCN.
        """
        B, N, _ = x.shape
        # --- Step 0: Classifier-free guidance logic
        cond_drop_prob = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob
        is_dropping_condition = np.random.choice([True, False], p=[cond_drop_prob, 1-cond_drop_prob])
        if is_dropping_condition:
            # (1) Convert obj_cond to zeros
            obj_cond = torch.zeros_like(obj_cond, device=x.device)
            # (2) Make edge_cond store a fully connected graph [2, B*N*N]
            edge_cond = self._create_combination_matrix(B, N, device=x.device)
            # (3) Set all relation_cond types to 'unknown' (0) (the length now matches edge_cond)
            relation_cond = torch.zeros_like(edge_cond[0], device=x.device)

        
        # --- Step 1: Unconditional denoising/diffusion
        x = self.attention_module(x)
        
        # --- Step 2: Inject the time embedding
        # adapt the time embedding shape ([B, F] -> [B, 1, F]) to use broadcasting when adding to fused_output [B, N, F]
        time_embedded = self.time_embedding_module(t)[:, None, :]
        x += time_embedded


        # --- Step 3: Scene graph processing
        graph_output = self.encoder_module(obj_cond, edge_cond, relation_cond)


        # --- Step 4: Instead of stacking [B, N, ...], RGCN uses [B*N, ...] approach, so we need to reshape X and fuse it with the graph_output
        x = x.view(B*N, -1)
        fused_output = torch.cat([x, graph_output], dim=-1)

        # --- Step 5: Final relational GCN
        # Note: to feed the data back to RGCN, we need to reshape the data back to [B*N, ...]
        output = self.fused_rgcn_module(
            fused_output,
            edge_cond, 
            relation_cond
        )

        # --- Step 6: Reshape the output back to [B, N, ...]
        output = output.view(B, N, -1)
        return output


    def forward_with_cond_scale(self, x, t, obj_cond, edge_cond, relation_cond, cond_scale):
        """
        Forward pass of the GuidedDiffusionNetwork with conditional scaling.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D] representing the initial input.
            t (torch.Tensor): Time tensor of shape [B] representing the time information.
            obj_cond (torch.Tensor): Object condition tensor of shape [B*N, C] representing the object conditions.
            edge_cond (torch.Tensor): Edge condition tensor of shape [2, E] representing the edge conditions.
            relation_cond (torch.Tensor): Relation condition tensor of shape [E] representing the relation conditions.
            cond_scale (float): Scaling factor for conditional loss.

        Returns:
            torch.Tensor: Scaled loss tensor.
        """
        cond_loss = self.forward(x, t, obj_cond, edge_cond, relation_cond, cond_drop_prob=0.)
        
        if cond_scale == 1:
            return cond_loss
        
        uncond_loss = self.forward(x, t, obj_cond, edge_cond, relation_cond, cond_drop_prob=1.)
        
        scaled_loss = uncond_loss + (cond_loss - uncond_loss) * cond_scale
        # TODO: add rescaled_phi here?
        return scaled_loss

    def _create_combination_matrix(self, B, N, device):
        """
        Create an edge connectivity combination matrix matching a fully connected graph.
        Used for classifier-free guidance when dropping the condition to avoid leaking connectivity information in the RGCN structure itself.

        Args:
            B (int): Batch size.
            N (int): Number of nodes.

        Returns:
            torch.Tensor: Combination matrix of shape [2, B*N*N] representing all possible combinations.
        """
        # Generate all possible combinations
        combinations = list(itertools.product(range(N), repeat=2))

        # Convert combinations to a PyTorch tensor
        combinations_tensor = torch.tensor(combinations, device=device).t()

        # Repeat the tensor B times along the second dimension
        repeated_combinations = combinations_tensor.repeat(1, B)

        return repeated_combinations
