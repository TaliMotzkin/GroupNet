import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch
import math



class TrajectoryDataset(Dataset):
    def __init__(self, past_traj, future_traj, selected_traj, H_list):
        """
        past_traj: Tensor (Total_samples, N, 5, 2)
        future_traj: Tensor (Total_samples, N, 10, 2, 20) - future trajectories per agent
        selected_traj: Tensor (Total_samples, N, 10, 2) - selected trajectory per agent
        H_list: Tensor (Total_samples, edges, N) - H matrix from model
        """
        self.past_traj = past_traj
        self.future_traj = future_traj
        self.selected_traj = selected_traj
        self.H_list = H_list

    def __len__(self):
        return self.past_traj.shape[0]

    def __getitem__(self, idx):
        return {
            'past_traj': self.past_traj[idx],
            'group_net': self.future_traj[idx],
            'selected_traj': self.selected_traj[idx],
            'H_list': self.H_list[idx]
        }


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

class FISHDatasetGAN(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, model, real_data, obs_len=5, pred_len=10, training=True,use_validation=False, validation_split=0.1
    ):
        super(FISHDatasetGAN, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if training:
            data_root = 'datasets/fish/fish/train_overlap.npy'
        else:
            data_root = 'datasets/fish/fish/test_overlap.npy'

        trajs = np.load(data_root)

        if not training and use_validation:

            split_idx = int((1 - validation_split) * len(trajs))
            trajs = trajs[split_idx:]


        self.trajs = trajs

        self.batch_len = len(self.trajs)
        print("batch_len" ,self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3) #number of traj, number of fish, number of time frames, xy coords
        print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        return out


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


