import numpy as np

import os
import sys
import argparse
import time
import numpy as np
import torch
import random
from torch import optim
from torch.optim import lr_scheduler
from model.encoder import MLPEncoder, MLP, MLP_fPIM, compute_alpha_im
from torch.utils.data import DataLoader
from data.dataloader_fish import FISHDataset2
from graphsZebra import edge_idx , fully_connected_graph
from utilities.utils import gumbel_softmax


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


def node2edge( x, rel_rec, rel_send):

    receivers = torch.matmul(rel_rec,   x)  # sums up the features of all nodes that receive edges, according to the graph's connectivity.
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([receivers, senders], dim=2)  # concatenates the features of the receiver and sender nodes
    return edges

def train(train_loader,epoch,  rel_rec, rel_send, mlp_model, f_PIM, tau):
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
        I_HG = gumbel_softmax(I_PIM, tau=tau, dim=0, hard=True)  # Hard sampling for binary values, what is the most probable group each i will be part of
        print(I_HG.sum(dim=-1))
        print("I_HG shape:", I_HG.shape)
        print(I_HG)

        alpha_im = compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send)

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
    train(train_loader,epoch, rel_rec, rel_send, mlp_model, f_PIM, tau)

    # """ save model """
    # if  (epoch + 1) % args.model_save_epoch == 0:
    #     model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,'model_cfg': args}
    #     saved_path = os.path.join(args.model_save_dir,str(epoch+1)+'.p')
    #     torch.save(model_saved, saved_path)