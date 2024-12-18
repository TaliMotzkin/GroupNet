import math
import torch
import random
import numpy as np
import itertools
import networkx as nx
import torch.nn as nn
import torch.nn.functional as Func
from torch.autograd import Variable
from utilities.utils import gumbel_softmax
# import torch_geometric
# import torch_geometric.nn as geom_nn
#
#
# gnn_layer_by_name = {
#     "GCN": geom_nn.GCNConv,
#     "GAT": geom_nn.GATConv,
#     "GraphConv": geom_nn.GraphConv
# }

class SeparateGRUs(nn.Module):
    def __init__(self, input_size1, hidden_size1, input_size2, hidden_size2,  num_layers=1):
        super(SeparateGRUs, self).__init__()
        self.gru1 = nn.GRU(input_size1, hidden_size1,num_layers)
        self.gru2 = nn.GRU(input_size2, hidden_size2,num_layers)

    def forward(self, x1, x2,  hidden1=None,  hidden2=None):
        # Process e_cg_2 through gru1
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)

        B = x1.size(1)  # Batch size for the current input

        # Adjust hidden1 if it exists but batch size doesn't match
        if hidden1 is not None:
            if hidden1.size(1) > B:
                hidden1 = hidden1[:, :B, :]  # Slice to match the current batch size

        if hidden2 is not None:
            if hidden2.size(1) > B:
                hidden2 = hidden2[:, :B, :]  # Slice to match the current batch size



        output1, h_n1 = self.gru1(x1, hidden1)
        #print(f"Input x1 shape: {x1.shape}")
        # Process e_HG_2 through gru2
        output2, h_n2 = self.gru2(x2, hidden2)
        #print(f"Output1 shape: {output1.shape}")  # Should be [seq_len, batch, hidden_size]
        #print(f"h_n1 shape: {h_n1.shape}")
        return (output1.permute(1, 0, 2), h_n1), (output2.permute(1, 0, 2), h_n2)


class RelationTypeInference(nn.Module):
    def __init__(self, edge_input_dim, hyperedge_input_dim, num_edge_types, num_hyperedge_types, tau=1.0):
        """
        Module to infer edge and hyperedge types then using the Gumbel-Softmax trick.

        Args:
            edge_input_dim: Dimension of edge feature embeddings e_CG^2.
            hyperedge_input_dim: Dimension of hyperedge feature embeddings e_HG^2.
            num_edge_types: Number of possible edge types (L_CG).
            num_hyperedge_types: Number of possible hyperedge types (L_HG).
            tau: Gumbel-Softmax temperature parameter.
        """
        super(RelationTypeInference, self).__init__()
        self.tau = tau

        # MLPs for edge logits
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim),
            nn.ReLU(),
            nn.Linear(edge_input_dim, num_edge_types)  # Output logits for edge types
        )

        # MLPs for hyperedge logits
        self.hyperedge_mlp = nn.Sequential(
            nn.Linear(hyperedge_input_dim, hyperedge_input_dim),
            nn.ReLU(),
            nn.Linear(hyperedge_input_dim, num_hyperedge_types)  # Output logits for hyperedge types
        )

    def forward(self, e_CG, e_HG):
        """
        Args:
            e_CG: Edge features [B, E, F] (Batch, Edges, Features).
            e_HG: Hyperedge features [B, M, F] (Batch, Hyperedges, Features).

        Returns:
            z_CG: Probabilities for edge types [B, E, L_CG].
            z_HG: Probabilities for hyperedge types [B, M, L_HG].
        """
        # Step 1: Compute logits for edge types
        edge_logits = self.edge_mlp(e_CG)  # [B, E, L_CG]

        # Step 2: Compute logits for hyperedge types
        hyperedge_logits = self.hyperedge_mlp(e_HG)  # [B, M, L_HG]


        return edge_logits, hyperedge_logits

class HyperEdgeAttention(nn.Module):
    def __init__(self, input_dim_e,input_dim_v, hidden_dim,node_dim, alpha=0.2):
        """
        claculates alpha_mi for nodes in hyperedges.

        Args:
            input_dim: Dimensionality of input features (F).
            hidden_dim: Dimensionality after linear transformations.
            alpha: Negative slope for LeakyReLU activation.
        """
        super().__init__()
        self.W1 = nn.Linear(input_dim_e, hidden_dim, bias=False)  # For e_HG
        self.W2 = nn.Linear(input_dim_v, hidden_dim, bias=False)  # For v_CG
        self.attention_vector = nn.Parameter(torch.Tensor(hidden_dim*2))  # Attention vector a
        self.leaky_relu = nn.LeakyReLU(alpha)

        # Initialize parameters
        nn.init.xavier_uniform_(self.W1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0), gain=1.414)

        # Node MLP (same for all heads)
        self.f_HG_v = nn.Sequential(
            nn.Linear(input_dim_e, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.BatchNorm1d(node_dim),
        )

        self.f_HG_2 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.BatchNorm1d(node_dim)
        )

    def forward(self, e_HG, v_CG, I_HG):
        """
        Args:
            e_HG: Hyperedge features, shape [B, M, F].
            v_CG: Node features, shape [B, N, F].
            I_HG: Incidence matrix, shape [B, N, M]. not sure if needed..

        Returns:
            alpha_mi: Attention weights between nodes and hyperedges, shape [B, N, M]. mybe transpose?
        """
        B, N, M = I_HG.shape
        F = e_HG.shape[-1]

        # Step 1: Transform features
        e_HG_proj = self.W1(e_HG)  # Shape: [B, M, hidden_dim]
        v_CG_proj = self.W2(v_CG)  # Shape: [B, N, hidden_dim]

        #print("e_HG_proj ", e_HG_proj.shape)
        #print("v_CG_proj", v_CG_proj.shape)
        # Step 2: Expand and combine features for attention
        e_HG_expanded = e_HG_proj.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, hidden_dim] N replicted
        v_CG_expanded = v_CG_proj.unsqueeze(2).expand(-1, -1, M, -1)  # [B, N, M, hidden_dim]

        # Concatenate and apply attention
        combined_features = torch.cat([e_HG_expanded, v_CG_expanded], dim=-1)  # [B, N, M, 2*hidden_dim]

        #print("combined_features ", combined_features.shape)
        attn_logits = self.leaky_relu(torch.einsum("bnmf,f->bnm", combined_features, self.attention_vector))  # [B, N, M]
        #print("attn_logits", attn_logits.shape)
        # print("attn_logits values", attn_logits)

        # Step 3: Mask logits using I_HG (nodes not in hyperedge -> mask)
        attn_logits = attn_logits.masked_fill(I_HG == 0, float('-inf'))
        # print("attn_logits masked ", attn_logits)

        # Step 4: Normalize attention scores
        alpha_mi = Func.softmax(attn_logits/100, dim=1)  # Softmax over nodes N #todo see if keep it 100
        alpha_mi = torch.nan_to_num(alpha_mi, nan=0.0).transpose(1,2)  # Replace NaNs with 0

        # print("alpha_mi", alpha_mi.shape)# Shape: [B, N, M]

        v_HG_1 = torch.einsum('bmn,bmf->bnf', alpha_mi, e_HG)  # Weighted aggregation: [B, N, F]

        #print("v_HG_1", v_HG_1.shape)
        # Apply f_HG,v (MLP) to v_HG^1
        v_HG_1_flat = v_HG_1.view(B * N, -1)  # Flatten for batchnorm
        v_HG_1 = self.f_HG_v(v_HG_1_flat).view(B, N, -1)  # [B, N, F]
        #print("v_HG_1", v_HG_1.shape)

        e_HG_2 = torch.einsum('bnm,bnf->bmf', I_HG, v_HG_1)  # Aggregate nodes to hyperedges

        # Apply f_HG^2 (MLP) to e_HG^2
        e_HG_2_flat = e_HG_2.view(B * M, -1)  # Flatten for batchnorm
        e_HG_2 = self.f_HG_2(e_HG_2_flat).view(B, M, -1)  # [B, M, F]

        #print("e_HG_2", e_HG_2.shape)

        return e_HG_2


class MLPHGE(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob):
        super(MLPHGE, self).__init__()
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

    def forward(self, alpha_im, V_CG):
        """
      Compute hyperedge features e_HG,m^1.

      Args:
          alpha_im: Node-hyperedge attention weights, shape [B, N, M].
          V_CG: Node features, shape [B, N, F].

      Returns:
          e_HG: Hyperedge features, shape [B, M, F].
  """
        # Step 1: Normalize alpha_im over nodes connected to each hyperedge
        alpha_sum =  alpha_im.sum(dim=1)   # Sum of alpha_im across nodes for each hyperedge ->the donomitor

        alpha_norm = alpha_im / (alpha_sum.unsqueeze(1) + 1e-8)  # Normalize alpha_im: [B, N, M] ->the numinetor
        #print("alpha_norm", alpha_norm.shape)
        # Step 2: Weight node features V_CG by normalized attention

        weighted_nodes = torch.einsum('bnm, bnf -> bmf', alpha_norm, V_CG)  # [B, M, F]
        #print("weighted_nodes: ", weighted_nodes.shape)
        # Step 3: Pass through edge MLP to obtain e_HG

        x = Func.elu(self.fc1(weighted_nodes))
        x = Func.dropout(x, self.dropout_prob, training=self.training)
        x = Func.elu(self.fc2(x))
        x = Func.dropout(x, self.dropout_prob, training=self.training)
        e_HG = self.fc3(x)
        # print(e_HG)
        return e_HG




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
    # print("edge_mask ", edge_mask.shape)
    # print("alpha_ij", alpha_ij.shape)

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
    # print("alpha_im 2 ", alpha_im)
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
        # self.bn =  nn.BatchNorm1d(self.out_dim)

        # Edge MLP (same for all heads)
        self.f_CG_e = nn.Sequential(
            nn.Linear(2 * self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        # Node MLP (same for all heads)
        self.f_CG_v = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )

        # Initialization
        nn.init.xavier_uniform_(self.projection.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_forward, gain=1.414)
        nn.init.xavier_uniform_(self.a_backward, gain=1.414)

    # def batch_norm(self, inputs):
    #     x = inputs.view(inputs.size(0) * inputs.size(1), -1)
    #     x = self.bn(x)
    #     return x.view(inputs.size(0), inputs.size(1), -1)

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

        #print("v_proj shape ", v_proj.shape)


        #print("rel_send ", rel_send.shape)


        # Step 3: Compute node features for each edge
        h_src = torch.einsum("ben,bnhd->behd", rel_send, v_proj)  # [B, E, H, D]
        h_tgt = torch.einsum("ben,bnhd->behd", rel_rec, v_proj)  # [B, E, H, D]

        #print("h_tgt ", h_tgt.shape)
        # Step 4: Compute attention scores for forward edges (i -> j)
        #todo make sure to look again at this tau dividings and maybe concatinate the two alphas..
        attn_ij = self.leaky_relu(torch.einsum("behd,hd->beh", h_src, self.a_forward))/500 # [B, E, num_heads]

        # Step 5: Compute attention scores for backward edges (j -> i)
        attn_ji = self.leaky_relu(torch.einsum("behd,hd->beh", h_tgt, self.a_backward))/500 # [B, E, num_heads]

        #print("attn_ji ", attn_ji.shape)
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

        #print("v_tgt ",v_tgt.shape)

        #Step 8:Compute edge features e_CG,ij^1
        weighted_v_src = alpha_ij.unsqueeze(-1) * v_src  # [B, E, H, D]
        weighted_v_tgt = alpha_ji.unsqueeze(-1) * v_tgt # [B, E, H, D]
        edge_input = torch.cat([weighted_v_src, weighted_v_tgt], dim=-1)  # [B, E, H, 2*D]
        e_CG= self.f_CG_e(edge_input.view(B, -1, 2 * D)).view(B, -1, H, D)  # [B, E, H, D]

        #print("e_CG", e_CG.shape) #B, E, H, D=128
        # Step 9: Aggregate edge features back to nodes
        edge_weighted = e_CG * alpha_ij.unsqueeze(-1)  # [B, E, H, D]

        #print("edge_weighted ", edge_weighted.shape)
        v_social = torch.einsum("behd,ben->bnhd", edge_weighted, rel_rec)  # [B, N, H, D]

        v_social = self.f_CG_v(v_social)
        #print("v_social before agg " ,v_social.shape)
        # Step 10: Combine heads
        if self.concat_heads:
            v_social = v_social.reshape(B, N, H * D)  # Concatenate heads
        else:
            v_social = v_social.mean(dim=2)  # Average heads

        #print("v_social after agg head ", v_social.shape)
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
        #print(f'MLP forward :{inputs.shape}')
        x = Func.elu(self.fc1(inputs))
        x = Func.dropout(x, self.dropout_prob, training=self.training)
        x = Func.elu(self.fc2(x))
        x = Func.dropout(x, self.dropout_prob, training=self.training)
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
        self.f_CG_v = MLP(num_heads * self.out_dim if concat_heads else self.out_dim, self.out_dim, n_out, do_prob)
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
        #print(f'edge2node: {rel_rec.shape} and x {x.shape}')
        incoming = torch.matmul(rel_rec, x)
        return incoming / incoming.size(1)

    def edge2node(self, x, rel_rec, rel_send):
        #print(f'node2edge encoder = {x.shape}')
        receivers = torch.matmul(rel_rec.transpose(1,2), x)
        senders = torch.matmul(rel_send.transpose(1,2), x)
        #print(f'sender: {receivers.shape}')
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        #inputs -  node features in a graph, 0 - batches, 1 number of nodes?
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        # B = x.shape[0]
        # rel_send = rel_send.expand(B, -1, -1)
        # rel_rec = rel_rec.expand(B, -1, -1)

        #print(f'this is X {x.shape}') # Batch size, nodes, time steps*features
        v_self = self.f_h(x)
        #print(f'this is v_self: {v_self.shape}')# [B, N, hidden_dim=16]
        v_social, alpha_ij = self.atten(v_self, rel_rec, rel_send) # [B, N, hidden_dim=16]

        #print(f'this is v_social: {v_social.shape}')
        # x = self.node2edge(x, rel_rec, rel_send)

        x = self.f_CG_v(v_social)
        # x_skip = x #skip connection
        #
        # if self.factor:
        #     x = self.node2edge(x, rel_rec, rel_send)
        #     #print(f'this is X after edge to node{x.shape}')
        #     x = self.mlp3(x)
        #     x = self.edge2node(x, rel_rec, rel_send)
        #     #print(f'this is X after node to edge 2 {x.shape}')
        #     x = torch.cat((x, x_skip), dim=2) #back and forth conversion?
        #     #print(f' x after skip connection{x.shape}')
        #     x = self.mlp4(x)
        # else:
        #     x = self.mlp3(x)
        #     x = torch.cat((x, x_skip), dim=2)
        #     x = self.mlp4(x)
        # return self.fc_out(x), v_self, alpha_ij
        return x, v_self, alpha_ij

