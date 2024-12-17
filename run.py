import torch
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

import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, x):
        output, h_n = self.gru(x)
        return output, h_n

# Define input dimensions
input_size = 5
hidden_size = 20
seq_len = 380
batch_size = 2

# Instantiate the model
model = SimpleGRU(input_size, hidden_size)

# Create a random input tensor [batch_size, seq_len, input_size]
x = torch.randn(batch_size, seq_len, input_size)

# Forward pass
output, h_n = model(x)

print("Output shape:", output.shape)  # Expected: [batch_size, seq_len, hidden_size]
print("Hidden state shape:", h_n.shape)  # Expected: [num_layers, batch_size, hidden_size]
