import pandas as pd
import numpy as np
# import torch
#
# # Number of nodes
# num_nodes = 3
#
# # Edges (source, target)
# edges = [(1, 2), (2, 3), (3, 1)]
#
# # Initialize matrices with zeros
# rel_send = torch.zeros(len(edges), num_nodes)
# rel_rec = torch.zeros(len(edges), num_nodes)
#
# # Fill the matrices
# for idx, (src, tgt) in enumerate(edges):
#     rel_send[idx, src - 1] = 1  # -1 because Python uses 0-indexing
#     rel_rec[idx, tgt - 1] = 1
#
# print("rel_send matrix:\n", rel_send)
# print("rel_rec matrix:\n", rel_rec)

# import torch
# import torch.nn as nn

# class SimpleGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(SimpleGRU, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers)
#
#     def forward(self, x):
#         output, h_n = self.gru(x)
#         return output, h_n
#
# # Define input dimensions
# input_size = 5
# hidden_size = 20
# seq_len = 380
# batch_size = 2
#
# # Instantiate the model
# model = SimpleGRU(input_size, hidden_size)
#
# # Create a random input tensor [batch_size, seq_len, input_size]
# x = torch.randn(batch_size, seq_len, input_size)
#
# # Forward pass
# output, h_n = model(x)
#
# print("Output shape:", output.shape)  # Expected: [batch_size, seq_len, hidden_size]
# print("Hidden state shape:", h_n.shape)  # Expected: [num_layers, batch_size, hidden_size]


#
# listik = [1,2,3,4,5]
# print(listik[-2])
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# scale = 3
# actor_number = 8
#
# if scale < actor_number:
#     group_size = scale
#     all_combs = []
#     for i in range(actor_number):  # or each actor i, generate all possible combinations of group_size - 1 other actors, excluding actor i
#         tensor_a = torch.arange(actor_number)  # [0,1,2...19]
#         # print(tensor_a)
#         # print(tensor_a[0:i], tensor_a[i + 1:])
#         tensor_a = torch.cat((tensor_a[0:i], tensor_a[i + 1:]), dim=0)  # all indx except of i's
#         # print(tensor_a)
#         padding = (1, 0, 0, 0)
#         all_comb = F.pad(torch.combinations(tensor_a, r=group_size - 1), padding,
#                          value=i)  # generate all combinations of group sized, if 3 -> [1,2,4]....
#
#         # print(all_comb[None, :, :].shape, all_comb)
#
#         all_combs.append(all_comb[None, :, :])  # A tensor of shape (1, C, group_size) containing all combinations of group_size actors, including actor i
#     # print(all_combs)
#     all_combs = torch.cat(all_combs, dim=0)
#     all_combs = all_combs # 8, 21, 3
#     # print(all_combs.shape)
#
# batch = 32
# ftraj_input = torch.from_numpy(np.random.rand(batch, 8, 10)) # batch, AGENTS, T* 2(dim, not xy)
#
#
# query_input = F.normalize(ftraj_input,p=2,dim=2)
# feat_corr = torch.matmul(query_input,query_input.permute(0,2,1)) # B, N, N
#
#
# all_indice = all_combs.clone() #(N,C,m) (actor_number, C, group_size)
# all_indice = all_indice[None,:,:,:].repeat(batch,1,1,1) # 32, N, C, s
#
# # print(all_indice.shape)
#
# all_matrix = feat_corr[:,None,None,:,:].repeat(1,actor_number,all_indice.shape[2],1,1)
# # print(all_matrix.shape) # B, N, C, N, N
# #all indices repeat:  # 32, N, C, s, N
#
# all_matrix = torch.gather(all_matrix,3,all_indice[:,:,:,:,None].repeat(1,1,1,1,actor_number))
#
# # print(all_matrix.shape) #torch.Size([32, 8, 21, 3, 8])
#
# #all indices repeat:  # 32, N, C, s, s
# all_matrix = torch.gather(all_matrix,4,all_indice[:,:,:,None,:].repeat(1,1,1,scale,1))
#
# # print(all_matrix.shape) # 32, N, C, s, s
#
# score = torch.sum(all_matrix,dim=(3,4),keepdim=False)
#
# # print(score.shape)
# _,max_idx = torch.max(score,dim=2)#coses the best combination!
#
# # print(max_idx.shape) #B, N
# indice = torch.gather(all_indice,2,max_idx[:,:,None,None].repeat(1,1,1,scale))[:,:,0,:]
# # print(max_idx[:,:,None,None].repeat(1,1,1,scale).shape) # B, N, 1, s
#
# # print(torch.gather(all_indice,2,max_idx[:,:,None,None].repeat(1,1,1,scale)).shape) # B, N, 1, s
#
# # print(indice.shape)
#
# H_matrix = torch.zeros(batch,actor_number,actor_number)
# H_matrix = H_matrix.scatter(2,indice,1)
# # print(H_matrix.shape)
#
#
# ##Other H matrix option!
# _,indice = torch.topk(feat_corr,dim=2,k=scale,largest=True) #For each actor, select the top group_size neighbors based on correlation.
# #indice A tensor of shape (batch_size, actor_number, group_size) containing indices of the top correlated actors.
# # print(indice.shape)
# H_matrix = torch.zeros(batch,actor_number,actor_number)
# H_matrix = H_matrix.scatter(2,indice,1)
# # print(H_matrix.shape)

# ftraj_input = torch.from_numpy(np.random.rand(32, 8, 10)) # batch, AGENTS, T* 2(dim, not xy)
#
# query_input = F.normalize(ftraj_input,p=2,dim=2)
# print(query_input.shape)
# feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))
# print(feat_corr.shape)
#
# query_input = torch.tensor([[[1.0, 2.0, 3.0],
#                              [4.0, 5.0, 6.0]]])
#
# # Permute to swap the last two dimensions, resulting in shape (1, 3, 2)
# permuted_input = query_input.permute(0, 2, 1)
#
# # Perform matrix multiplication resulting in a 2x2 correlation matrix
# feat_corr = torch.matmul(query_input, permuted_input)
#
# # Print the outputs
# print("Original Input:", query_input)
# print("Permuted Input:", permuted_input)
# print("Feature Correlation Matrix:", feat_corr)
#
# batch = 2
# actor_number = 4
#
# # Create a zero matrix
# H_matrix = torch.zeros(batch, actor_number, actor_number)
#
# # Assume some indices (this should be created based on your specific needs)
# indice = torch.tensor([[[1, 2], [0, 1]], [[3, 0], [2, 1]]])  # Example indices
# print(indice.shape) #2, 2, 2
# # Scatter 1's into H_matrix according to indices
# H_matrix = H_matrix.scatter(2, indice, 1)
#
# print(H_matrix)





# import torch
# import torch.nn.functional as F
# np.random.seed(0)
# torch.random.manual_seed(0)
#
#
#
# def calculate_softmax_loss(pred, target, model_output):
#     # Calculate squared Euclidean distances
#     # pred shape: [B*N, S, T, 2]
#     # target shape: [B*N, T, 2]
#     target_expanded = target.unsqueeze(1).expand_as(pred)  # [B*N, S, T, 2]
#     # print(target_expanded)
#     diff = pred - target_expanded  # Difference
#     print("diff", diff)
#     dist_squared = diff.pow(2).sum(dim=-1).sum(dim=-1)  # Sum squared differences across T and 2 dimensions
#     print("dist_squared", dist_squared.shape, dist_squared)
#     soft_targets = F.softmax(-dist_squared, dim=1) #larger distnace - smaller values
#     print("soft_targets", soft_targets)
#
#     soft_targets = torch.tensor([[0.8, 0.02, 0.08, 0.1],
#             [0.1, 0.02, 0.8, 0.08],
#             [0.07, 0.9, 0.01, 0.02]])
#
#     probabilities = model_output
#     print("probabilities", probabilities)
#     predicted_probs = torch.clamp(probabilities, min=1e-9)
#
#     kl_div = soft_targets * torch.log(soft_targets / predicted_probs)
#     print("kl_div", kl_div)
#
#     entropy = -torch.sum(predicted_probs * torch.log(predicted_probs), dim=1).mean()
#     final_loss = torch.sum(kl_div, dim=1).mean()
#     print("entropy", entropy)
#     print("final_loss", final_loss)
#     print("minus entorpy",final_loss+entropy )
#
#
#     return final_loss
#
#
#
# target = torch.rand(3, 5, 2)
# pred = torch.rand(3,4, 5, 2)
# prob = torch.tensor([[0.99, 0.001, 0.001, 0.008],[ 0.001, 0.001, 0.99, 0.008],[0.001, 0.99,0.001, 0.008]])
# target = torch.tensor([[[1, 1],
#          [1, 1],
#          [2, 2],
#          [2, 2],
#          [2,2]],
#
#         [[0, 0],
#          [0., 0],
#          [0., 0],
#          [0., 0.],
#          [0., 0.]],
#
#         [[0.3, 0.3],
#          [0.3, 0.3],
#          [0.3, 0.3],
#          [0.3, 0.3],
#          [0.3, 0.3]]])
# pred = torch.tensor([[[[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]],
#
#          [[0., 0.2437],
#           [0., 0.],
#           [0., 0.],
#           [0.8155, 0.7932],
#           [0.2783, 0.4820]],
#
#          [[0.8198, 0.9971],
#           [0.6984, 0.5675],
#           [0.8352, 0.2056],
#           [0.5932, 0.1123],
#           [0.1535, 0.2417]],
#
#          [[0.7262, 0.7011],
#           [0.2038, 0.6511],
#           [0.7745, 0.4369],
#           [0.5191, 0.6159],
#           [0.8102, 0.9801]]],
#
#
#         [[[0., 0.],
#           [0., 0.],
#           [0., 0.],
#           [0., 0.],
#           [0., 0.]],
#
#          [[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]],
#
#          [[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]],
#
#          [[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]]],
#
#
#         [[[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]],
#
#          [[0.3, 0.3],
#           [0.3, 0.3],
#           [0.3, 0.3],
#           [0.3, 0.3],
#           [0.3, 0.3]],
#
#          [[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]],
#
#          [[1, 1],
#           [1, 1],
#           [2, 2],
#           [2, 2],
#           [2, 2]]]])
# prob = torch.tensor([[0.001, 0.99, 0.001, 0.008],[ 0.001, 0.001, 0.008, 0.99],[0.001, 0.001, 0.99, 0.008]])
# prob = torch.tensor([[0.2, 0.2, 0.2, 0.4],[0.2, 0.2, 0.2, 0.4],[0.2, 0.2, 0.2, 0.4]])
# prob = torch.tensor([[0.4, 0.2, 0.2, 0.2],[0.2, 0.2, 0.4, 0.2],[0.2, 0.4, 0.2, 0.2]])

# print("target", target)
# calculate_softmax_loss(pred, target, prob)
# print(pred)


###############@#@##@

# import torch

# Example dimensions
# B, N, S, T, XY = 1, 2, 3, 5, 2  # (B*N, S, T, XY)
# BN = B * N  # Total batch size
#
# # Example input tensor (random data for demonstration)
# future_traj = torch.randn(BN, T, XY)  # (B*N, T, 2)
# diverse_pred_traj = torch.randn(BN, S, T, XY)  # (B*N, S, T, 2)
#
#
# # Expand `future_traj` to match `diverse_pred_traj`
# target_expanded = future_traj.unsqueeze(1).expand_as(diverse_pred_traj)  # (B*N, S, T, 2)
# print("target_expanded", diverse_pred_traj)
#
#
# # Reshape each S trajectory to flatten (T,2) → (T*2)
# target_flattened = diverse_pred_traj.reshape(BN, S*T*XY)  # (B*N, S*T*2)
# print("target_flattened", target_flattened)
#
# # Concatenate all S options together along the last dimension
# target_concat = target_flattened.unsqueeze(1).expand(BN, S, S*T*XY).reshape(BN*S, S*T*XY)  # (B*N, S, T*2*S)
# print("target_concat", target_concat)
#
# serial_numbers = torch.arange(S).repeat(BN).unsqueeze(-1) # (B*N, S, 1)
#
# # Step 4: Concatenate the serial number to the trajectory data
# final_output = torch.cat((target_concat, serial_numbers), dim=-1) # (B*N, S, (T*XY*S) + 1)
#
# # Print final shape
# print(final_output)
#
#
#
#
#
# B, N, S = 1, 2, 3  # (B*N, S)
# BN = B * N  # Total batch size

# Example class labels (random values for demonstration)
# class_labels = torch.randint(0, S, (BN, S))  # (B*N, S), simulated ranking class labels
# print(class_labels, "class_labels")
# # Step 1: Repeat each row S times → (BN*S, S)
# class_labels_expanded = class_labels.repeat_interleave(S, dim=0)  # (B*N*S, S)
#
# # Print final shape
# print(class_labels_expanded)


data = pd.read_csv("xgb_training_data.csv")
print(data[0])