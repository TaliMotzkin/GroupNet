import numpy as np

import os
import sys
import argparse
import time
import numpy as np
import torch.nn as nn

import torch
import random
from torch import optim
from torch.optim import lr_scheduler
from model.encoder import MLPEncoder, MLP, MLP_fPIM, compute_alpha_im, MLPHGE, HyperEdgeAttention, RelationTypeInference
from torch.utils.data import DataLoader
from model.decoder import RNNDecoder
from data.dataloader_fish import FISHDataset2
from graphsZebra import edge_idx , fully_connected_graph
from utilities.utils import gumbel_softmax
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures reproducibility on the GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
print('device:',device)


def build_dynamic_graph_and_hypergraph(z_CG, z_HG, rel_rec, rel_send, I_HG):
    """
    Builds dynamic rel_rec, rel_send, and I_HG matrices based on inferred edge/hyperedge types.

    Args:
        z_CG: Edge type probabilities, shape [B, E, L_CG].
        z_HG: Hyperedge type probabilities, shape [B, M, L_HG].
        rel_rec: Original receiver matrix [B, E, N].
        rel_send: Original sender matrix [B, E, N].
        I_HG: Original incidence matrix [B, N, M].

    Returns:
        new_rel_rec: Updated receiver matrix [B, E', N].
        new_rel_send: Updated sender matrix [B, E', N].
        new_I_HG: Updated incidence matrix [B, N, M'].
        edge_types: Edge types [B, E'].
        hyperedge_types: Hyperedge types [B, M'].
    """
    B, E, L_CG = z_CG.shape  # Batch size, number of edges, edge types
    B, M, L_HG = z_HG.shape  # Batch size, number of hyperedges, hyperedge types
    N = rel_rec.shape[-1]  # Number of nodes

    rel_send = rel_send.expand(B, -1, -1)  # Shape becomes [B, E, N]
    rel_rec = rel_rec.expand(B, -1, -1)

    # Step 1: Identify the most probable edge and hyperedge types
    edge_types = z_CG.argmax(dim=-1)  # [B, E]
    hyperedge_types = z_HG.argmax(dim=-1)  # [B, M]
    # print("edge_types", edge_types) #the argument of the edge type

    # Step 2: Filter valid edges and hyperedges (ignore "no edge" type = 0)
    valid_edges_mask = edge_types != 0  # [B, E]
    valid_hyperedges_mask = hyperedge_types != 0  # [B, M]
    # print("valid_hyperedges_mask", valid_hyperedges_mask) #true false maskes

    # print("I_HG", I_HG)
    # Initialize new graph/hypergraph matrices with zeros
    new_rel_rec = torch.zeros_like(rel_rec)  # [B, E, N]
    new_rel_send = torch.zeros_like(rel_send)  # [B, E, N]
    new_I_HG = torch.zeros_like(I_HG)  # [B, N, M]

    # Populate valid edges and hyperedges
    for b in range(B):  # Iterate over batches
        # Valid edges
        valid_edges = valid_edges_mask[b].nonzero(as_tuple=True)[0]  # Indices of valid edges
        new_rel_rec[b, valid_edges] = rel_rec[b, valid_edges]  # Preserve valid receiver nodes
        new_rel_send[b, valid_edges] = rel_send[b, valid_edges]  # Preserve valid sender nodes

        # Valid hyperedges
        valid_hyperedges = valid_hyperedges_mask[b].nonzero(as_tuple=True)[0]  # Indices of valid hyperedges
        # print(valid_hyperedges,"valid_hyperedges")
        new_I_HG[b, :, valid_hyperedges] = I_HG[b, :, valid_hyperedges]  # Preserve valid node-hyperedge relations

    return new_rel_rec, new_rel_send, new_I_HG, edge_types, hyperedge_types



class SeparateGRUs(nn.Module):
    def __init__(self, input_size1, hidden_size1, input_size2, hidden_size2,  num_layers=1):
        super(SeparateGRUs, self).__init__()
        self.gru1 = nn.GRU(input_size1, hidden_size1,num_layers)
        self.gru2 = nn.GRU(input_size2, hidden_size2,num_layers)

    def forward(self, x1, x2,  hidden1=None,  hidden2=None):
        # Process e_cg_2 through gru1
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        output1, h_n1 = self.gru1(x1, hidden1)
        print(f"Input x1 shape: {x1.shape}")
        # Process e_HG_2 through gru2
        output2, h_n2 = self.gru2(x2, hidden2)
        print(f"Output1 shape: {output1.shape}")  # Should be [seq_len, batch, hidden_size]
        print(f"h_n1 shape: {h_n1.shape}")
        return (output1.permute(1, 0, 2), h_n1), (output2.permute(1, 0, 2), h_n2)

def node2edge( x, rel_rec, rel_send):

    receivers = torch.matmul(rel_rec,   x)  # sums up the features of all nodes that receive edges, according to the graph's connectivity.
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([receivers, senders], dim=2)  # concatenates the features of the receiver and sender nodes
    # print("EDGED", edges.shape)
    return edges

def train(train_loader,epoch,  rel_rec, rel_send, mlp_model, f_PIM, tau, f_HG_E,attention_hyper,L_infer, gru_model, decoder):
    set_seed(42)
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        # print(data)
        v_social, v_self, alpha_ij = model(data,  rel_rec, rel_send)
        # B, N, T, F =v_social.shape
        print("v_social shape: ",v_social.shape)
        # v_social = v_social.view(B, N, T, -1).to(device)
        # print(v_self[0])
        print("v_self shape: ",v_self.shape)
        v_combined = torch.cat([v_self, v_social], dim=-1)  # Shape: [B, N, 2+hidden_dim/10]
        print("v_combined shape: ",v_combined.shape)

        print("edges shape EC2: ", node2edge(v_combined, rel_rec, rel_send).shape)
        e_cg_2 = mlp_model(node2edge(v_combined, rel_rec, rel_send))
        print("e_cg_2 shape: " , e_cg_2.shape)


        """ The other route for the hypergraph"""
        I_PIM  = f_PIM(v_combined)
        # print(v_combined)

        print("I_PIM: ", I_PIM.shape)
        I_HG = gumbel_softmax(I_PIM, tau=tau, dim=-1, hard=True)  # Hard sampling for binary values, what is the most probable group each i will be part of
        print("I_HG shape:", I_HG.shape)
        # print(I_HG)

        alpha_im = compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send)
        print("alpha_im", alpha_im.shape)

        e_HG = f_HG_E(alpha_im, v_combined)
        print("e_HG", e_HG.shape)

        e_HG_2 = attention_hyper(e_HG, v_combined, I_HG)

        """ Getting the edges types"""
        (edge_logits, h_g), (hyperedge_logits, h_hg) = gru_model(e_cg_2,e_HG_2) #todo pass this hidden layers to the next batch?

        # print("output1", output1)
        print("edge_logits", edge_logits.shape)

        # edge_logits, hyperedge_logits =L_infer(e_cg_2,e_HG_2)
        z_CG  = gumbel_softmax(edge_logits, tau=tau, dim=-1, hard=False)
        z_HG = gumbel_softmax(hyperedge_logits, tau=tau, dim=-1, hard=False)
        # print("Edge type probabilities (z_CG):", z_CG)  # [B, E, F]
        print("Hyperedge type probabilities (z_HG):", z_HG.shape)  # [B, M, F]
        print("z_CG", z_CG.shape)

        #todo consider printing the infered graphs
        new_rel_rec, new_rel_send, new_I_HG, new_edge_types, new_hyperedge_types = build_dynamic_graph_and_hypergraph(z_CG, z_HG, rel_rec, rel_send, I_HG)
        #maybe only for visualizations!!

        # print("new_rel_rec", new_rel_rec.shape)
        # print("new_rel_rec", new_rel_rec)
        # print("new_edge_types", new_edge_types)

        rel_rec = new_rel_rec
        rel_send = new_rel_send
        I_HG = new_I_HG


        """ Decoding """
        decoder(data, z_CG, rel_rec, rel_send, z_HG, I_HG, 10, v_combined)


        #first type = no edge!
    #     """ optimize """
    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()
    #
    #     if iter_num % args.iternum_print == 0:
    #         print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f}'
    #         .format(epoch,args.num_epochs,iter_num,total_iter_num,total_loss.item(),loss_pred,loss_recover,loss_kl,loss_diverse))
    #     iter_num += 1
    #
    # scheduler.step()
    # model.step_annealer()


''' arguments'''

lr = 1e-4
decay_step =10
decay_gamma = 0.5
batch_size = 32
num_epochs = 1
n_hid = 128
n_out = 5
n_in_mlp = (n_hid+n_out)*2
tau = 1
do_prob = 0.2
Ledge = 30
Lhyper = 10

""" model & optimizer """
set_seed(42)
model = MLPEncoder( num_heads = 1, n_in = 10, n_hid = n_hid, n_out = n_out) #n_in = 5 (trajectory)*xy(2) =10
mlp_model = MLP(n_in_mlp, n_hid, n_out)
# print("model mlp",mlp_model)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

""" dataloader """
train_set = FISHDataset2(training=True)

agents_number = train_set.agents_num()
# M = int(agents_number / 2)
M = 5
f_PIM = MLP_fPIM(n_hid+n_out, M)
f_HG_E = MLPHGE(n_hid+n_out, n_hid, n_out*3, do_prob)
attention_hyper = HyperEdgeAttention(n_out*3,n_hid+n_out, n_hid, n_out*5)
gru_model = SeparateGRUs(n_out, Ledge, n_out*5, Lhyper)
L_infer = RelationTypeInference(n_out, n_out*5, Ledge, Lhyper)
decoder = RNNDecoder(n_in_mlp,n_out,2, Ledge, Lhyper,n_hid)
# decoder=2
# print("agents ", agents_number)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True)

""" start training """
edge = fully_connected_graph(agents_number)
rel_rec, rel_send = edge_idx(edge,agents_number )
rel_rec =rel_rec.unsqueeze(0)
rel_send =rel_send.unsqueeze(0)

model.to(device)
for epoch in range(0, num_epochs):
    train(train_loader,epoch, rel_rec, rel_send, mlp_model, f_PIM, tau, f_HG_E,attention_hyper,L_infer, gru_model, decoder)

    # """ save model """
    # if  (epoch + 1) % args.model_save_epoch == 0:
    #     model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,'model_cfg': args}
    #     saved_path = os.path.join(args.model_save_dir,str(epoch+1)+'.p')
    #     torch.save(model_saved, saved_path)