import itertools
import numpy as np
import torch
import torch.nn as nn

from attention_layer import SelfMultiheadAttention
from attention_layer import CrossMultiheadAttention
from relational_gcn import RelationalRGCN
from time_embedding import TimeEmbedding

class GuidedDiffusionNetwork(nn.Module):
    def __init__(
        self,
        layer_1_dim,
        layer_2_dim,
        general_params,
        attention_params,
        rgc_params,
        cond_drop_prob
        ):
        super(GuidedDiffusionNetwork, self).__init__()
        
        self.cond_drop_prob = cond_drop_prob
        
        self.time = GuidedDiffusionTime(
            dim_t = 14
        )
        
        self.rgc1 = GuidedDiffusionRGC(
            layer_dim=layer_2_dim,
            general_params=general_params,
            attention_params=attention_params,
            rgc_params=rgc_params
        )
        
        self.block1 = GuidedDiffusionBlock(
            layer_dim=layer_2_dim,
            general_params=general_params,
            attention_params=attention_params,
            rgc_params=rgc_params
        )
        
        self.linear1 = nn.Linear(
            in_features=layer_2_dim,
            out_features=layer_2_dim
        )
            
        self.rgc2 = GuidedDiffusionRGC(
            layer_dim=layer_2_dim,
            general_params=general_params,
            attention_params=attention_params,
            rgc_params=rgc_params
        )
        
        self.linear2 = nn.Linear(
            in_features=layer_2_dim,
            out_features=layer_2_dim
        )
            
        self.rgc3 = GuidedDiffusionRGC(
            layer_dim=layer_2_dim,
            general_params=general_params,
            attention_params=attention_params,
            rgc_params=rgc_params

        )
        
        
        self.linear3 = nn.Linear(
            in_features=2*layer_2_dim,
            out_features=layer_2_dim
        )
        
        self.linear4 = nn.Linear(
            in_features=2*layer_2_dim,
            out_features=layer_1_dim
        )
        
        assert general_params["obj_cond_dim"] % layer_1_dim == 0, "Layer 1 dim needs to be a divisor of obj cond dim"
        #assert general_params["obj_cond_dim"] % layer_2_dim == 0, "Layer 2 dim needs to be a divisor of obj cond dim"
        
    def forward(self, x, t, obj_cond, edge_cond_in, relation_cond_in, cond_drop_prob=None):
        """
        Forward pass of the GuidedDiffusionNetwork.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D] representing the initial input.
            t (torch.Tensor): Time tensor of shape [B] representing the corresponding timesteps.
            obj_cond (torch.Tensor): Object condition tensor of shape [B, N, C] representing the object condition.
            edge_cond (torch.Tensor): Edge condition tensor of shape [2, E] representing the edge condition.
            relation_cond (torch.Tensor): Relation condition tensor of shape [E] representing the corresponding relation type condition.
            cond_drop_prob (float, optional): Probability of dropping the classifier-free guidance. If none is provided, the default model's probability is used.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, D] representing the predicted noise of the final fused relational GCN.
        """
        
        B, N, _ = x.shape
        cond_drop_prob = 1  #AAAAAACHTUNG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # --- Step 0: Classifier-free guidance logic
        cond_drop_prob = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob
        is_dropping_condition = np.random.choice([True, False], p=[cond_drop_prob, 1-cond_drop_prob])
        if is_dropping_condition:
            # (1) Convert obj_cond to zeros
            # TODO: remove me when it's clear what to do with that
            #obj_cond = torch.zeros_like(obj_cond, device=x.device)  #DOPPELACHTUNG!!!!!
            # (2) Make edge_cond store a fully connected graph [2, B*N*N]
            edge_all = self._create_combination_matrix(B, N, device=x.device)
            # (3) Set all relation_cond types to 'unknown' (0) (the length now matches edge_cond)
            relation_all = torch.zeros_like(edge_all[0], device=x.device)
        
        edge_cond = torch.cat((edge_all, edge_cond_in), dim=-1)
        relation_cond = torch.cat((relation_all, relation_cond_in), dim=-1)
            
        # --- Step 1: Block1
        # x_1 = self.block1(x, t, obj_cond, edge_cond, relation_cond) 
        
        # # --- Step 2: Linear Layer + Activation
        # x_2 = self.linear1(x_1)
        # x_3 = torch.tanh(x_2)
        
        # # --- Step 3: Block2
        # x_4 = self.block2(x_3, t, obj_cond, edge_cond, relation_cond) 
        
        # # --- Step 4: Linear Layer + Activation
        # x_5 = self.linear2(x_4)
        # x_6 = torch.tanh(x_5)
        
        # # --- Step 5: Skip Connection, Block 3
        # x_7 = x_6 + x_1
        # x_8 = self.block3(x_7, t, obj_cond, edge_cond, relation_cond)
        
        # # --- Step 6: Linear Layer + Activation
        # x_9 = self.linear3(x_8)
        # output = torch.tanh(x_9)
        
        output1 = self.time(x, t)
        output1 = self.linear1(output1)
        output1 = nn.Tanh()(output1)
        
        output2 = self.rgc1(output1, t, obj_cond, edge_cond, relation_cond)
        
        output3a = self.rgc2(output2, t, obj_cond, edge_cond, relation_cond)
        
        output3 = self.block1(output3a, t, obj_cond, edge_cond, relation_cond)
        
        output4a = self.linear3(output3)
        output4a = nn.Tanh()(output4a)
        
        output4 = self.rgc3(output4a, t, obj_cond, edge_cond, relation_cond)
        
        output5 = torch.cat((output4, output1), dim=-1)
        output = self.linear4(output5)
            
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
        
        # For every batch step i, add the offset of N*i to both rows window size of N*N
        for i in range(1, B+1):
            l_step, r_step = N*N*i, N*N*(i+1)
            repeated_combinations[:, l_step:r_step] += N*i

        return repeated_combinations
   

class GuidedDiffusionTime(nn.Module):
    def __init__(
        self,
        dim_t
    ):
        super(GuidedDiffusionTime, self).__init__()
        
        self.time_embedding_module = TimeEmbedding(dim=dim_t)
        
    def forward(self, x, t):
        
        B, N, D = x.shape
        
        # --- Step 1: Injecting time and label embeddings
        time_embedded = self.time_embedding_module(t)
        time_embedded = time_embedded.unsqueeze(1).expand(B, N, -1)
        
        #time_embedded = t.unsqueeze(1).unsqueeze(2)
        #time_embedded = time_embedded.expand(B, N, 1)

        x_t = torch.cat((x, time_embedded), dim=-1)
        
        return x_t

class GuidedDiffusionRGC(nn.Module):
    def __init__(
        self,
        layer_dim,
        general_params,
        attention_params,
        rgc_params
    ):
        super(GuidedDiffusionRGC, self).__init__()
        
        # Instantiate the activation functions from the string
        if rgc_params["rgc_activation"] == "tanh":
            rgc_activation = nn.Tanh()
        elif rgc_params["rgc_activation"] == "leakyrelu":
            rgc_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif rgc_params["rgc_activation"] == "relu":
            rgc_activation = nn.ReLU(inplace=True)
        elif rgc_params["rgc_activation"] == "silu":
            rgc_activation = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError(f"Activation function {rgc_activation} is not implemented.")
            
        # Instantiate hidden_dims tuples from the string
        rgc_hidden_dims = eval(rgc_params["rgc_hidden_dims"])
        
        self.rgc_module = RelationalRGCN(
            in_channels=layer_dim, 
            h_channels_dims=rgc_hidden_dims,
            out_channels=layer_dim,
            num_relations=rgc_params["rgc_num_relations"], 
            num_bases=rgc_params["rgc_num_bases"],  
            aggr=rgc_params["rgc_aggr"],
            activation=rgc_activation,
            dp_rate=rgc_params["rgc_dp_rate"],
            bias=rgc_params["rgc_bias"],
        )
        
    def forward(self, x, t, obj_cond, edge_cond, relation_cond):
        
        B, N, D = x.shape
        
        x = x.view(B*N, -1)
        rgcn_out = self.rgc_module(
            x,
            edge_cond, 
            relation_cond
        )
        rgcn_out = rgcn_out.view(B, N, -1)
        
        return rgcn_out
    

class GuidedDiffusionBlock(nn.Module):
    def __init__(
        self,
        layer_dim,
        general_params,
        attention_params,
        rgc_params
    ):
        super(GuidedDiffusionBlock, self).__init__()
        
        # Instantiate the activation functions from the string
        if rgc_params["rgc_activation"] == "tanh":
            rgc_activation = nn.Tanh()
        elif rgc_params["rgc_activation"] == "leakyrelu":
            rgc_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif rgc_params["rgc_activation"] == "relu":
            rgc_activation = nn.ReLU(inplace=True)
        elif rgc_params["rgc_activation"] == "silu":
            rgc_activation = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError(f"Activation function {rgc_activation} is not implemented.")
            
        # Instantiate hidden_dims tuples from the string
        rgc_hidden_dims = eval(rgc_params["rgc_hidden_dims"])
        kernel_size = ((general_params["obj_cond_dim"]//25),) # HARDCODED
        
        self.time_embedding_module = TimeEmbedding(dim=14) # ACHTUNG HARDCODED AKTUELL
        
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)
        
        self.rgc_module = RelationalRGCN(
            in_channels=layer_dim, 
            h_channels_dims=rgc_hidden_dims,
            out_channels=layer_dim,
            num_relations=rgc_params["rgc_num_relations"], 
            num_bases=rgc_params["rgc_num_bases"],  
            aggr=rgc_params["rgc_aggr"],
            activation=rgc_activation,
            dp_rate=rgc_params["rgc_dp_rate"],
            bias=rgc_params["rgc_bias"],
        )
        
        self.self_attention_module = SelfMultiheadAttention(
            N=general_params["num_obj"], 
            D=layer_dim, 
            embed_dim=attention_params["attention_self_head_dim"]*attention_params["attention_num_heads"], 
            num_heads=attention_params["attention_num_heads"]
        )
        
        self.cross_attention_module = CrossMultiheadAttention(
            N=general_params["num_obj"], 
            D=layer_dim, 
            C=general_params["obj_cond_dim"], 
            embed_dim=attention_params["attention_cross_head_dim"]*attention_params["attention_num_heads"], 
            num_heads=attention_params["attention_num_heads"]
        )
        
    def forward(self, x, t, obj_cond, edge_cond, relation_cond):
        """
        Forward pass of one Guided Diffusion Block

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D] representing the initial input.
            t (torch.Tensor): Time tensor of shape [B] representing the corresponding timesteps.
            obj_cond (torch.Tensor): Object condition tensor of shape [B, N, C] representing the object condition.
            edge_cond (torch.Tensor): Edge condition tensor of shape [2, E] representing the edge condition.
            relation_cond (torch.Tensor): Relation condition tensor of shape [E] representing the corresponding relation type condition.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, D] representing the predicted noise of the final fused relational GCN.
        """
        
        B, N, D = x.shape
        
        # --- Step 1: Injecting time and label embeddings
        #time_embedded = self.time_embedding_module(t)
        #time_embedded = time_embedded.unsqueeze(1).expand(B, N, -1)

        #x_t = torch.cat((x, time_embedded), dim=-1)
        
        obj_cond_pooled = self.max_pool(obj_cond)
        x_t_text = x + torch.cat((obj_cond_pooled, torch.zeros(B, N, 4)), dim=-1)
        
        # # --- Step 2: Relational GCN processing
        # x_t_text = x_t_text.view(B*N, -1)
        #x = x.view(B*N, -1)
        #rgcn_out = self.rgc_module(
        #    x,
        #    edge_cond, 
        #    relation_cond
        #)
        #rgcn_out = rgcn_out.view(B, N, -1)
        
        # --- Step 3a: Self-Attention
        self_out = self.self_attention_module(x)
        
        # # --- Step 3b: Cross-Attention
        cross_out = self.cross_attention_module(x, obj_cond)
        
        # # --- Step 4: Sum up Parallel Attention Paths
        att = self_out + cross_out
        output = torch.cat((x_t_text, att), dim=-1)
        
        return output
        