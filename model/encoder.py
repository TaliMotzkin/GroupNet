import math
import torch
import random
import numpy as np
import itertools
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils.utils import gumbel_softmax, custom_softmax, sample_core
# import torch_geometric
# import torch_geometric.nn as geom_nn
#
#
# gnn_layer_by_name = {
#     "GCN": geom_nn.GCNConv,
#     "GAT": geom_nn.GATConv,
#     "GraphConv": geom_nn.GraphConv
# }

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch




def compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send):
    """
    Compute alpha_im for each node-hyperedge pair.
    Args:
        alpha_ij: Pairwise attention scores, shape [B, E].
        I_HG: Incidence matrix for hypergraph, shape [B, N, M].
        rel_rec: Receiver mask, shape [B, E, N].
        rel_send: Sender mask, shape [B, E, N].
    Returns:
        alpha_im: Node-hyperedge attention weights, shape [B, N, M].
    """
    B, N, M = I_HG.shape
    E = alpha_ij.shape[1]

    # print(alpha_ij)
    # Step 1: Expand dimensions for masking
    I_HG_expanded = I_HG.unsqueeze(1)  # [B, 1, N, M]
    rel_rec_expanded = rel_rec.unsqueeze(-1)  # [B, E, N, 1]
    rel_send_expanded = rel_send.unsqueeze(-1)  # [B, E, N, 1]

    # Step 2: Determine if an edge belongs to a hyperedge
    edge_mask_rec = (rel_rec_expanded * I_HG_expanded).sum(2) > 0  # [B, E, M] checkes if node i is a reciver and also set to 1 in the hg matrix
    edge_mask_send = (rel_send_expanded * I_HG_expanded).sum(2) > 0  # [B, E, M]
    edge_mask = edge_mask_rec & edge_mask_send  # Both sender and receiver must be in hyperedge
    print("edge_mask ", edge_mask.shape)
    print("alpha_ij", alpha_ij.shape)

    # Step 3: Mask and sum alpha_ij for edges belonging to each hyperedge
    alpha_ij_masked = alpha_ij * edge_mask  # Mask edges for each hyperedge # B, E, M

    # print("alpha_ij_masked ", alpha_ij_masked)
    # For every node, sum the attention scores of edges connected to the node within each hyperedge
    alpha_im = torch.einsum("bem,ben->bnm", alpha_ij_masked, rel_rec)  # Aggregate attention scores

    # print("alpha_im ", alpha_im.shape)
    # print("alpha_im ", alpha_im)

    # Step 4: Normalize by the number of nodes in each hyperedge
    N_H_m = I_HG.sum(dim=1, keepdim=True)  # Number of nodes per hyperedge [B, 1, M]
    # print(N_H_m)
    alpha_im = alpha_im / (N_H_m - 1 + 1e-8)  # Avoid division by zero
    print("alpha_im 2 ", alpha_im)
    return alpha_im  # Shape: [B, N, M]





# Define the MLP for f_PIM
class MLP_fPIM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10):
        """
        Multi-Layer Perceptron for computing I_PIM (probabilistic incidence matrix).

        Args:
            input_dim: Dimensionality of input features (F)
            output_dim: Number of hyperedges (M)
            hidden_dim: Size of hidden layers
        """
        super(MLP_fPIM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output dimension: Number of hyperedges M
        )

    def forward(self, x):
        return self.net(x)  # Shape: [B, N, M]


class TemporalGATLayer(nn.Module):
    def __init__(self,out_dim, input_dim, hidden_dim, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Temporal Graph Attention Layer with Multi-Head Attention.

        Args:
            input_dim: Dimensionality of input features (F).
            hidden_dim: Dimensionality of output features per head.
            num_heads: Number of attention heads.
            concat_heads: If True, concatenate outputs of all heads; otherwise, average.
            alpha: Negative slope for LeakyReLU in attention computation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.out_dim = out_dim




        # Attention-related learnable parameters
        self.projection = nn.Linear(hidden_dim, self.out_dim * num_heads, bias=False)
        self.a_forward = nn.Parameter(torch.Tensor(num_heads, self.out_dim))
        self.a_backward  = nn.Parameter(torch.Tensor(num_heads, self.out_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)

        # Edge MLP (same for all heads)
        self.f_CG_e = nn.Sequential(
            nn.Linear(2 * self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        # Initialization
        nn.init.xavier_uniform_(self.projection.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_forward, gain=1.414)
        nn.init.xavier_uniform_(self.a_backward, gain=1.414)

    def forward(self, v_self, rel_rec, rel_send):
        """
        Args:
            x: Input node features of shape [B, N, T, F] (batch, nodes, time, features).
            rel_rec: Receiver adjacency matrix [E, N].
            rel_send: Sender adjacency matrix [E, N].

        Returns:
            Updated node features: [B, N, hidden_dim].
        """
        B, N, F = v_self.shape
        H = self.num_heads
        D = self.out_dim

        # Step 2: Apply linear transformation to compute projections
        v_proj = self.projection(v_self)# [B, N, H*out_dim]
        v_proj = v_proj.view(B, N, H, D)  # [B, N, H, D]

        print("v_proj shape ", v_proj.shape)


        print("rel_send ", rel_send.shape)


        # Step 3: Compute node features for each edge
        h_src = torch.einsum("ben,bnhd->behd", rel_send, v_proj)  # [B, E, H, D]
        h_tgt = torch.einsum("ben,bnhd->behd", rel_rec, v_proj)  # [B, E, H, D]

        print("h_tgt ", h_tgt.shape)
        # Step 4: Compute attention scores for forward edges (i -> j)
        #todo make sure to look again at this tau dividings and maybe concatinate the two alphas..
        attn_ij = self.leaky_relu(torch.einsum("behd,hd->beh", h_src, self.a_forward))/500 # [B, E, num_heads]

        # Step 5: Compute attention scores for backward edges (j -> i)
        attn_ji = self.leaky_relu(torch.einsum("behd,hd->beh", h_tgt, self.a_backward))/500 # [B, E, num_heads]

        print("attn_ji ", attn_ji.shape)
        attn_max = torch.maximum(attn_ij, attn_ji)
        attn_ij_stable = torch.exp(attn_ij - attn_max)
        attn_ji_stable = torch.exp(attn_ji - attn_max)

        # Step 6: Normalize attention scores
        alpha_sum = torch.exp(attn_ij_stable) + torch.exp(attn_ji_stable)  # Sum of forward and backward scores
        alpha_ij = torch.exp(attn_ij_stable) / alpha_sum  # Normalized attention for forward edge
        alpha_ji = torch.exp(attn_ji_stable) / alpha_sum  # Normalized attention for backward edge  # [B, E, H]
        # print(attn_ij)
        # print(attn_max)
        # print(attn_ij_stable)

        # print(alpha_sum.shape, "alpha")

        # Step 7: Weight originalv_proj with normalized attention
        v_src = torch.einsum("ben,bnhd->behd", rel_send, v_proj) # [B, E, H, D]
        v_tgt = torch.einsum("ben,bnhd->behd", rel_rec, v_proj)  # [B, E, H, D]

        print("v_tgt ",v_tgt.shape)

        #Step 8:Compute edge features e_CG,ij^1
        weighted_v_src = alpha_ij.unsqueeze(-1) * v_src  # [B, E, H, D]
        weighted_v_tgt = alpha_ji.unsqueeze(-1) * v_tgt # [B, E, H, D]
        edge_input = torch.cat([weighted_v_src, weighted_v_tgt], dim=-1)  # [B, E, H, 2*D]
        e_CG= self.f_CG_e(edge_input.view(B, -1, 2 * D)).view(B, -1, H, D)  # [B, E, H, D]

        print("e_CG", e_CG.shape)
        # Step 9: Aggregate edge features back to nodes
        edge_weighted = e_CG * alpha_ij.unsqueeze(-1)  # [B, E, H, D]

        print("edge_weighted ", edge_weighted.shape)
        v_social = torch.einsum("behd,ben->bnhd", edge_weighted, rel_rec)  # [B, N, H, D]

        print("v_social before agg " ,v_social.shape)
        # Step 10: Combine heads
        if self.concat_heads:
            v_social = v_social.reshape(B, N, H * D)  # Concatenate heads
        else:
            v_social = v_social.mean(dim=2)  # Average heads

        print("v_social after agg head ", v_social.shape)
        # print(v_social)
        return v_social, alpha_ij



class MLP(nn.Module):
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        print(f'MLP forward :{inputs.shape}')
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.fc3(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(self, num_heads, n_in, n_hid, n_out, do_prob=0.0, factor=True, concat_heads =True ):
        super(MLPEncoder, self).__init__()
        self.factor = factor
        self.f_h = MLP(n_in, n_hid, n_hid, do_prob)
        # self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        if concat_heads:
            assert n_hid % num_heads == 0, "hidden_dim must be divisible by num_heads"
            self.out_dim = n_hid
        else:
            self.out_dim = n_hid // num_heads

        # Final node MLP
        self.f_CG_v = MLP(num_heads * self.out_dim if concat_heads else self.out_dim, self.out_dim, self.out_dim, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3 if factor else n_hid * 2, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()
        self.atten = TemporalGATLayer(out_dim = self.out_dim, input_dim = n_in, hidden_dim = n_hid,  num_heads = num_heads, concat_heads = concat_heads)
        self.mlp5 = MLP(n_hid, n_hid, n_hid, do_prob)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, rel_rec, rel_send):
        print(f'edge2node: {rel_rec.shape} and x {x.shape}')
        incoming = torch.matmul(rel_rec, x)
        return incoming / incoming.size(1)

    def edge2node(self, x, rel_rec, rel_send):
        print(f'node2edge encoder = {x.shape}')
        receivers = torch.matmul(rel_rec.transpose(1,2), x)
        senders = torch.matmul(rel_send.transpose(1,2), x)
        print(f'sender: {receivers.shape}')
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        #inputs -  node features in a graph, 0 - batches, 1 number of nodes?
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        B = x.shape[0]
        rel_send = rel_send.expand(B, -1, -1)
        rel_rec = rel_rec.expand(B, -1, -1)

        print(f'this is X {x.shape}') # Batch size, nodes, time steps*features
        v_self = self.f_h(x)
        print(f'this is v_self: {v_self.shape}')# [B, N, hidden_dim=16]
        v_social, alpha_ij = self.atten(v_self, rel_rec, rel_send) # [B, N, hidden_dim=16]

        print(f'this is v_social: {v_social.shape}')
        # x = self.node2edge(x, rel_rec, rel_send)

        x = self.f_CG_v(v_social)
        x_skip = x #skip connection

        if self.factor:
            x = self.node2edge(x, rel_rec, rel_send)
            print(f'this is X after edge to node{x.shape}')
            x = self.mlp3(x)
            x = self.edge2node(x, rel_rec, rel_send)
            print(f'this is X after node to edge 2 {x.shape}')
            x = torch.cat((x, x_skip), dim=2) #back and forth conversion?
            print(f' x after skip connection{x.shape}')
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
        return self.fc_out(x), v_self, alpha_ij
