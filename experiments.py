import numpy as np
import time
import os
import sys
import argparse
import time
import numpy as np
import torch.nn as nn
import math
from model.HGNN_model_fish import HGNNModelFish
import torch
import random
from torch import optim
from torch.optim import lr_scheduler
from model.encoder import MLPEncoder, MLP, MLP_fPIM, compute_alpha_im, MLPHGE, HyperEdgeAttention, RelationTypeInference
from torch.utils.data import DataLoader
from model.decoder import RNNDecoder
from data.dataloader_fish import FISHDataset2, seq_collate
from data.dataloader_nba import NBADataset
from graphsZebra import edge_idx , fully_connected_graph
from utilities.utils import gumbel_softmax, reconstruction_loss
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



def train( train_loader, epoch, rel_rec, rel_send, model, encoder_timesteps, recompute_gap, total_pred_steps,tau):
    starting_epoch_time = time.time()
    set_seed(42)
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    h_hg = None
    h_g = None
    I_HG = None
    print(epoch , "epoch")
    for data in train_loader:
        past_traj = data['past_traj'].to(device)  # Shape: [batch_size, num_fish, obs_len, 2]
        future_traj = data['future_traj'].to(device)

        output_lists, h_g, h_hg, rel_rec1, rel_send1, L_SM, L_SH, L_SP, L_KL = model(past_traj, total_pred_steps,
                                                                                   encoder_timesteps, recompute_gap,
                                                                                   rel_rec, rel_send, tau, h_g, h_hg)

        h_g = h_g.detach() if h_g is not None else None  # for not breaking the computational graph!
        h_hg = h_hg.detach() if h_hg is not None else None
        # print(len(output_lists))
        #
        # iter_num = 0
        # for sample in output_lists["trajs"]:
        #     #print("iter_num:",iter_num)
        #     print(sample[0][:3])
        #     iter_num += 1

        pred_trajs = torch.cat(output_lists["trajs"], dim=2)  # Concatenate along time (dim=2)
        # prin("Concatenated predictions shape:", pred_trajs.shape)
        # prin("future, ", future_traj.shape)

        #comparing predictions with ground truth
        # L_Rec = F.mse_loss(pred_trajs, future_traj, reduction='mean')  # Mean Squared Error
        # print("Reconstruction Loss (L_Rec):", L_Rec.item())

        means = torch.cat(output_lists["mus"], dim=2).mean(dim=3)
        # L_Rec_2 = F.mse_loss(future_traj, means)
        B, A, T, F = future_traj.shape
        L_Rec_2 = (future_traj - means).pow(2).sum()/ (B*T)

        # print("future", future_traj[0,0])
        # print("pasts", past_traj[0,0])
        # print("means ", means[0,0])

        if iter_num % 10 == 0:
            print("Reconstruction Loss 2 (L_Rec):", L_Rec_2.item())
            print("L_SM", L_SM.item())
            print("L_SH", L_SH.item())
            print("L_SP", L_SP.item())
            print("L_KL", L_KL.item())
            print(iter_num, "iterations")
        # if trust_predictions:
        #     inputs = torch.cat((history_inputs[:, :, -(T_h - T_p):, :], output_traj), dim=2)
        # else:
        #     inputs = torch.cat((history_inputs[:, :, -(T_h - T_p):, :], ground_truth_next_tp), dim=2)
        #
        """ optimize """
        optimizer.zero_grad()
        total_loss = L_Rec_2 + L_SM + L_SH + L_SP + L_KL
        if iter_num % 10 == 0:
            print("total_loss", total_loss.item())
            end_time = time.time()
            elapsed_time = end_time - starting_epoch_time
            print(f"Elapsed time for iteration {iter_num}: {elapsed_time:.2f} seconds")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # for name, parameter in model.named_parameters():
        #     if parameter.grad is not None:
        #         print(f"{name} gradient norm: {parameter.grad.norm().item()}")
        iter_num += 1
    scheduler.step()
    loss_list = [L_SM.item(), L_SH.item(), L_SP.item(), L_KL.item(), L_Rec_2.item(), total_loss.item()]
    return loss_list


        # if iter_num % iternum_print == 0:
        #         print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f}'
        #         .format(epoch,args.num_epochs,iter_num,total_iter_num,total_loss.item(),loss_pred,loss_recover,loss_kl,loss_diverse))
        #     iter_num += 1




''' arguments'''

lr = 0.001
n_in = 10
decay_step =10
decay_gamma = 0.85
batch_size = 32
num_epochs = 10
n_hid = 128
n_out = 5
tau = 1
n_head=1
do_prob = 0.2 #todo check this
Ledge = 3
Lhyper = 3
num_cores= 3
encoder_timesteps = 5
recompute_gap = 5
total_pred_steps=15
M=5
hard =False
model_save_epoch =10
model_save_dir = 'saved_models/nba/experiments'
""" model & optimizer """
set_seed(42)

model = HGNNModelFish(n_in, n_head,  n_out, n_hid,  M, Ledge, Lhyper,  num_cores, tau, hard,device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

""" dataloader and others """
# train_set = FISHDataset2(encoder_timesteps, recompute_gap, total_pred_steps,training=True)
train_set = NBADataset(obs_len=5,
    pred_len=10,
    training=True)
# agents_number = train_set.agents_num()
agents_number = 11

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=seq_collate,
    pin_memory=True)





""" start training """
model.to(device)
for epoch in range(0, num_epochs):
    edge = fully_connected_graph(agents_number)
    rel_rec, rel_send = edge_idx(edge, agents_number)
    rel_rec = rel_rec.unsqueeze(0).to(device)
    rel_send = rel_send.unsqueeze(0).to(device)
    loss = train(train_loader, epoch, rel_rec, rel_send, model, encoder_timesteps, recompute_gap, total_pred_steps, tau)

    """ save model """
    if (epoch + 1) % model_save_epoch == 0:
        model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'loss': loss}
        saved_path = os.path.join(model_save_dir, str(epoch + 1) + '.p')
        print("model saved")
        torch.save(model_saved, saved_path)