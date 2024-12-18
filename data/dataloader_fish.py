import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch
import math

def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'nba',
    }

    return data

class FISHDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):
        super(FISHDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if training:
            data_root = 'datasets/fish/fish/train.npy'
        else:
            data_root = 'datasets/fish/fish/test.npy'

        self.trajs = np.load(data_root)

        if training:
            self.trajs = self.trajs #84
        else:
            self.trajs = self.trajs #46

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3) #number of traj, number of fish, number of time frames, xy coords
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        return out


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@





class FISHDataset2(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, encoder_timesteps, recompute_gap, total_pred_steps,training=True,

    ):
        super(FISHDataset2, self).__init__()
        self.times = math.ceil((total_pred_steps - encoder_timesteps) / recompute_gap)
        self.encoder_timesteps = encoder_timesteps


        if training:
            data_root = 'datasets/fish/fish/train2.npy'
        else:
            data_root = 'datasets/fish/fish/test2.npy'

        self.trajs = np.load(data_root)

        if training:
            self.trajs = self.trajs #2535
        else:
            self.trajs = self.trajs #1365

        self.batch_len = len(self.trajs)
        print("length of data: ", self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3) #number of traj, number of fish, number of time frames, xy coords
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def agents_num(self):
        return self.traj_abs.shape[1]

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.encoder_timesteps, :]
        future_traj = self.traj_abs[index, :, self.encoder_timesteps:, :]
        out = [past_traj, future_traj]
        return out

        return out