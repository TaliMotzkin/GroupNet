import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


#load
file_path = './trajectories.csv'
trajectories_data = pd.read_csv(file_path)

class FishTrajectoryDataset(Dataset):
    def __init__(self, data, transpose=False):
        """
        Initializes the dataset.
        Args:
            data (pd.DataFrame): DataFrame containing the fish trajectory data.
            fframes per seconds-  29 for 1 second.make sure all have the same... (0.034), 19500 lines.
        """

        self.transpose = transpose
        self.data = self.interpolate_nans(data)
        # Group data by fish ID (xN, yN pairs) and reshape into batches
        self.fish_ids = [col.split('x')[1] for col in data.columns if col.startswith('x')]
        self.data = {fish_id: self.data[[f'x{fish_id}', f'y{fish_id}']].values for fish_id in self.fish_ids}
        self.num_frames = len(data)

        #for padding if necesery
        self.max_length = max(len(self.data[fish_id]) for fish_id in self.fish_ids)

    def __len__(self):
        #number of fishes
        return len(self.fish_ids)

    def get_trajectories(self):
        traj_num = self.num_frames // 150   #50 for about 2 seconds of past traj, 100 frames for 4 seconds prediction for future traj = 50+100= 150, 19500/150 = 130 traj total
        #for now it is set to one csv, maybe will need to add more from other datasets and manage the concatinations
        all_all_fish_loc = [] #shape (N,15, 20, 2) ->number of trajectories, frames per 1 trj, fish, x.y coords
        for i in range(traj_num):
            all_fish_loc = [] #shape 15, 20, 2)
            for j in range(15):
                time_stamp = 150 * i + 10 * j
                curr_frame = [self.data[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)# preparing 130 different trajectories
        all_all_fish_loc = np.array(all_all_fish_loc, dtype=np.float32)
        return all_all_fish_loc


    def getSlice(self, to_idx, start_idx=0):
        '''
        slicing a trajectory of all of the fish
        Args:
            to_idx:
            start_idx:

        Returns:

        '''

        results = []
        for i in self.fish_ids:
            slice_frame = self.data[i][start_idx:to_idx]
            results.append(slice_frame)
        results = torch.tensor(results, dtype=torch.float32)


        return results.permute(1, 0, 2)


    def interpolate_nans(self,data):
        """
        Changes NaN values in a data for each column independently.
        Returns:
            pd.DataFrame: data with NaNs replaced by interpolated values.
        """
        return data.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

    def pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum length with the padding value.
        Args:
            sequence (ndarray): Array of shape (frames, 2).
        Returns:
            ndarray: Padded array of shape (max_length, 2).
        """
        padded_sequence = np.zeros((self.max_length, 2))
        padded_sequence[:sequence.shape[0], :] = sequence
        #usingthe last valid value to pad the remaining rows
        if sequence.shape[0] > 0:
            padded_sequence[sequence.shape[0]:, :] = sequence[-1, :]
        return padded_sequence

    def __getitem__(self, idx):
        """
        Retrieves a fishs data .
        Args:
            idx (int): Index of the fish.
        Returns:
            torch.Tensor: tensor shape (max_length, 2), where 2 represents x and y coordinates.
        """

        fish_id = self.fish_ids[idx]
        trajectory = self.data[fish_id]
        padded_trj = self.pad_sequence(trajectory)

        if self.transpose:
            padded_trj = padded_trj.T

        return torch.tensor(padded_trj, dtype=torch.float32)

data_target = 'fish'
dataset = FishTrajectoryDataset(trajectories_data, True)
all_trajs = dataset.get_trajectories()
print(len(all_trajs))
index = list(range(len(all_trajs)))
from random import shuffle
shuffle(index)
train_set = all_trajs[index[:84]]
test_set = all_trajs[index[84:]]
print('train num:',train_set.shape[0])
print('test num:',test_set.shape[0])

np.save(data_target+'/train.npy',train_set)
np.save(data_target+'/test.npy',test_set)