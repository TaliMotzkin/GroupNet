from torch_geometric.data import Data
import torch
from Data_zebra import FishTrajectoryDataset
import pandas as pd

file_path = './trajectories.csv'
trajectories_data = pd.read_csv(file_path)
dataset = FishTrajectoryDataset(trajectories_data)
def construct_graph(dataset,  t, T_h=1):
    """
    Constructs a fully connected observation graph Gobs from fish trajectory data.
    Args:
        dataset (FishTrajectoryDataset): Dataset containing fish trajectory data.
        t (int): Current time step.
        T_h (int): Time window width.
    Returns:
        torch_geometric.data.Data: Fully connected graph representation of the fish data.
    """
    ##fish (nodes)
    num_fish = len(dataset)

    #the start and end of the time window
    start_time = max(0, t - T_h + 1)
    end_time = t + 1

    #extracts data for a certain time frame
    node_features = []
    for idx in range(num_fish):
        trajectory = dataset[idx]  # Shape: (frames, 2) or (2, frames) if transposed
        time_window_data = trajectory[start_time:end_time]  # Extract time window
        node_features.append(time_window_data)

    node_features = torch.stack(node_features)  # Shape: (N, T_h, 2)

    # Define edges (fully connected graph)
    edge_index = torch.combinations(torch.arange(num_fish), r=2, with_replacement=False).t() #create all possible conections between 2 nodes

    #for example- Original: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    #Transposed: [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]]

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Directed edges

    # Edge attributes: pairwise distances between fish over time
    edge_features = []
    for i, j in edge_index.T:
        distance = torch.norm(node_features[i] - node_features[j], dim=-1)  # Euclidean distance
        edge_features.append(distance)  # Aggregate distance over time

    # Stack edge features and adjust shape
    edge_features = torch.stack(edge_features)  # Shape: (E, T_h)
    edge_features = edge_features.unsqueeze(-1)  # Shape: (E, T_h, 1), if needed

    # Social attributes: aggregate trajectories for each node
    social_attributes = torch.zeros_like(node_features) #shape N, Th, 2
    for i in range(num_fish):
        others = torch.cat([node_features[j].unsqueeze(0) for j in range(num_fish) if j != i], dim=0)
        social_attributes[i] = others.mean(dim=0)  # Aggregate mean of others (T_h, 2)

    # Final node features: concatenate self and social attributes
    node_features = torch.cat([node_features, social_attributes], dim=-1)  # Shape: (N, frames, 4)

    # Create the PyTorch Geometric graph
    graph = Data(
        x=node_features,  # Node attributes
        edge_index=edge_index,  # Edge indices
        edge_attr=edge_features  # Edge attributes
    )
    return graph

# graph = construct_graph(dataset, 0)
# print(graph)


#Or making an rec+sender:

def fully_connected_graph(num_nodes):
    """
    Constructs a fully connected graph's edge index.
    Args:
        num_nodes (int): Number of nodes in the graph.
    Returns:
        torch.Tensor: Edge index of shape (2, E), where E = num_nodes * (num_nodes - 1).
    """
    edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).t()
    # Add reverse edges to make it directed
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def edge_idx(edge_index, num_nodes):
    """
    Constructs the rel_rec and rel_send matrices for a graph.
    Args:
        edge_index (torch.Tensor): Edge indices of shape (2, E), where E is the number of edges and num_nodes is the number of conecting nodes
        num_nodes (int): Total number of nodes in the graph.
    Returns:
        rel_rec (torch.Tensor): Relation receiver matrix of shape (E, N).
        rel_send (torch.Tensor): Relation sender matrix of shape (E, N).
    """
    num_edges = edge_index.shape[1]  # Number of edges

    # rel_rec and rel_send as zero matrices
    rel_rec = torch.zeros((num_edges, num_nodes), dtype=torch.float32)
    rel_send = torch.zeros((num_edges, num_nodes), dtype=torch.float32)

    #fill rel_rec and rel_send
    for edge_idx, (sender, receiver) in enumerate(edge_index.t()):
        rel_rec[edge_idx, receiver] = 1  # Mark the receiver node
        rel_send[edge_idx, sender] = 1   # Mark the sender node

    return rel_rec, rel_send



def rel_rec_rel_send_hypergraph(node_positions, threshold):
    """
    Constructs rel_rec and rel_send matrices for a hypergraph based on node groups.
    Args:
        node_positions (torch.Tensor): Positions of nodes in the first frame (N, 2).
        threshold (float): Distance threshold for grouping nodes into hyperedges.
    Returns:
        rel_rec (torch.Tensor): Relation receiver matrix of shape (E, N).
        rel_send (torch.Tensor): Relation sender matrix of shape (E, N).
    """
    num_nodes = node_positions.shape[0]
    groups = []  # List of hyperedges (groups of nodes)

    #make groups based on distance threshold
    for i in range(num_nodes):
        group = [i]
        for j in range(num_nodes):
            if i != j and torch.norm(node_positions[i] - node_positions[j]) < threshold:
                group.append(j)
        groups.append(list(set(group)))  #remove duplicates

    #remove duplicate hyperedges (convert to unique sets)
    unique_groups = [list(g) for g in {tuple(sorted(group)) for group in groups}]

    # Create rel_rec and rel_send matrices
    num_hyperedges = len(unique_groups)
    rel_rec = torch.zeros((num_hyperedges, num_nodes), dtype=torch.float32)
    rel_send = torch.zeros((num_hyperedges, num_nodes), dtype=torch.float32)

    for edge_idx, group in enumerate(unique_groups):
        for node in group:
            rel_rec[edge_idx, node] = 1  # all nodes are "receivers" ans senders
            rel_send[edge_idx, node] = 1

    return rel_rec, rel_send, unique_groups

num_fish = len(dataset)
edge = fully_connected_graph(num_fish)
rel_rec, rel_send = edge_idx(edge,num_fish )
# print(rel_rec.shape, rel_send)