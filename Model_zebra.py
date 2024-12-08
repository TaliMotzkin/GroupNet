from torch import nn
from encoder import MLPEncoder
from torch_geometric.data import Data
import torch
from Data_zebra import FishTrajectoryDataset
import pandas as pd
from graphsZebra import edge_idx , fully_connected_graph




#fish data  is fish id, trajectory, features
class ModelZebra(nn.Module):
    def __init__(self, n_input, n_hidden, n_out,):
        super(ModelZebra, self).__init__()
        self.n_input = n_input
        self.n_hidden =n_hidden
        self.n_out = n_out


    def forward(self, inputs, rel_rec1, rel_send1):
        encoder1 = MLPEncoder(self.n_input, self.n_hidden, self.n_out)
        graph = encoder1.forward(inputs, rel_rec1, rel_send1)
        # graph[:, :, :] = graph[0, 0] #take the output only of the first two nodes
        print(graph)


# file_path = './trajectories.csv'
# trajectories_data = pd.read_csv(file_path)
# dataset = FishTrajectoryDataset(trajectories_data)
# sliced = dataset.getSlice(1)
sliced = torch.tensor([[[0,1], [2,1], [0,3]]], dtype=torch.float32)

# sliced = sliced.permute(1,2,0)
# sliced = sliced.squeeze(2)
print(f'sliced:{sliced.shape}')

num_fish = len(sliced[0])
edge = fully_connected_graph(num_fish)
rel_rec, rel_send = edge_idx(edge,num_fish )
rel_rec =rel_rec.unsqueeze(0)
rel_send =rel_send.unsqueeze(0)


model = ModelZebra(2, 5, 2)
graph = model.forward(sliced, rel_rec, rel_send)
