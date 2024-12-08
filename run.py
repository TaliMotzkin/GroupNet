import torch

# Number of nodes
num_nodes = 3

# Edges (source, target)
edges = [(1, 2), (2, 3), (3, 1)]

# Initialize matrices with zeros
rel_send = torch.zeros(len(edges), num_nodes)
rel_rec = torch.zeros(len(edges), num_nodes)

# Fill the matrices
for idx, (src, tgt) in enumerate(edges):
    rel_send[idx, src - 1] = 1  # -1 because Python uses 0-indexing
    rel_rec[idx, tgt - 1] = 1

print("rel_send matrix:\n", rel_send)
print("rel_rec matrix:\n", rel_rec)
