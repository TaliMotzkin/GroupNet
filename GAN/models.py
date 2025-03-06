import os

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, e_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, e_dim).float() #512, 32
        position = torch.arange(0, max_len).unsqueeze(1) #This creates a column vector [0, 1, 2, ..., 511] of shape (512, 1).

        ## Shape: (16,)
        div_term = 10000.0 ** (torch.arange(0., e_dim, 2.) / e_dim) #torch.arange(0., e_dim, 2.) gives [0, 2, 4, ..., 30] → 16 values. The division e_dim=32 ensures different frequencies across dimensions

        # Calculate sin for even-numbered digits, and calculate cos for odd-numbered digits.
        pe[:, 0::2] = torch.sin(position / div_term) #Even indices (0, 2, 4, ..., 30) are assigned sine values
        pe[:, 1::2] = torch.cos(position / div_term) #Odd indices (1, 3, 5, ..., 31) are assigned cosine values

        pe = pe.unsqueeze(0).unsqueeze(0) ## Shape: (1, 1, 512, 32)
        # self.pe = pe
        self.register_buffer('pe', pe) #ensures pe is part of the model but doesn't get updated

    def forward(self, x):
        x = x + self.pe[:, :, : x.size(2)] #in pe extracts the first seq=5 positions → shape (1, 5, 32) Broadcasting adds it to x (shape (agents, 5, 32)).
        #maybe instead x = x + self.pe[:, : x.size(1)].detach() since Variable is from older version
        return self.dropout(x)


class Generator(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, noise_dim, traj_len, dropout, num_edges):
        super(Generator, self).__init__()
        self.cat_pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )
        self.pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )  # social
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim,  nhead=heads,dim_feedforward=mlp_dim,dropout=dropout)
            , num_layers=depth)  # Time series feature extraction

        self.space_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim+2,  nhead=heads,dim_feedforward=mlp_dim,dropout=dropout)
            , num_layers=depth)  # Spatial sequence feature extraction


        self.encoder = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=dim + 2, nhead=heads),
            nn.Linear(dim+2, dim),
            nn.ReLU()
        )

        self.pos_encoder = PositionalEncoding(dim)
        self.noise_dim = noise_dim

        self.edge_embedding = nn.Sequential(
            nn.Linear(num_edges, dim),
            nn.ReLU()
        )


        self.final_mlp = nn.Sequential(
            nn.Linear(dim+2+self.noise_dim+1, 2),
            nn.ReLU()
        )
        self.len = traj_len

        self.future_encoder = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )

        self.selection_net = nn.Linear(10*(dim+2), 1)
    #groupnet data is 20, B*N, T, 2 ----> ([20, 1024, 10, 2]) for future
    #H is B, Edges (different scales), N ---> ([128, 9, 8]) for past
    # past traj B, N, 5, 2 ----> 128,8, 5, 2
    #future traj B, N, 10, 2 ----> 128, 8, 10, 2
    #target xy ---> [1,2] for future
    #controlled_agent ---> [x] x is [0-n] for future
    #mission ---> [B, 0 or 1] for future

    def forward(self,Group_net_traj, H , past_traj, mission , controlled_agent, target, direction=None, who_I_see =None):

        np.random.seed(0)
        #node features past data timing
        #todo adding also features of who i see in my FOV, direction
        #Aggregation time features
        # traj_cat = torch.concat([ direction, who_I_see], dim=-1)  # B, number of agents, 5 seq, F?
        traj_cat = past_traj
        pos_emb = self.cat_pos_to_dim(traj_cat)  # B, N, 5, 32

        t_in = self.pos_encoder(pos_emb) # B, N, 5, 32

        # print("t_in", t_in.shape)
        t_in = t_in.view(t_in.shape[0]*t_in.shape[1],t_in.shape[2],-1)#bn, 5 , f
        t_in = t_in.permute(1, 0, 2)
        # print("t", t_in.shape)
        # print("past_traj.size(1)", past_traj.size(1), t_in.shape)
        mask = nn.Transformer().generate_square_subsequent_mask(t_in.shape[0]).to(past_traj)


        t_out = self.time_encoder(t_in, mask=mask)  # Time feature output #BN, 5, F
        t_out = t_out.permute(1,0,2)

        #group dynamics
        edge_feat = self.edge_embedding(H.permute(0, 2, 1)) #B, N, F
        edge_node_features = torch.cat([past_traj, edge_feat.unsqueeze(2).expand(-1, -1, 5, -1)], dim=-1)  # (B, N, 5, F+2)
        edge_node_features = edge_node_features.view(edge_node_features.shape[0]*edge_node_features.shape[1],edge_node_features.shape[2], -1)  # Flatten time for transformer input

        edge_node_features = edge_node_features.permute(1, 0, 2)
        # print("edge_node_features", edge_node_features.shape)
        past_rel_features = self.encoder(edge_node_features) #learns relationships over time #B*N, 5, F
        past_rel_features = past_rel_features.permute(1, 0, 2)

        # print("t_out", t_out.shape, past_rel_features.shape)
        #combine timeing with group dynamics
        past_rel_timed = t_out * past_rel_features  # (B*N, 5, F) #represents group interactions with a time-aware context

        #future dynamics with groupnet
        future_encoded = self.future_encoder(Group_net_traj.permute(1, 0, 2, 3))


        past_rel_expanded  = past_rel_timed.unsqueeze(1).expand(-1, 20, -1, -1)  # Repeat past time dynamics for each option # (B*N, 20, 5, F)
        target = torch.tensor(target, dtype=torch.float32, device= past_rel_timed.device)
        target = target.unsqueeze(0).expand(past_rel_timed.shape[0], -1)  # (B*N, 2)
        # print("target", target.shape)
        target_expanded = target.unsqueeze(1).unsqueeze(1).expand(-1, 20, 15, -1)  # (B*N, 20, 15, 2)
        # print("target_expanded", target_expanded.shape)

        combined_features = torch.cat([
            past_rel_expanded,  # (B*N, 20, 5, F)
            future_encoded  # (B*N, 20, 10, F)
        ], dim=2)

        # print("combined_features", combined_features[0])
        #Combined Past - Future(Freeze Gradients  #
        combined_features = combined_features.view(combined_features.shape[0] * 20, 15, -1).permute(1, 0, 2)  # (15, B*N*20, F)
        t_out_features = self.time_encoder(combined_features.detach()) # (15, B*N*20, F)

        t_out_features = torch.cat([
            t_out_features,  # (15, B*N*20, F)
            target_expanded.view(target_expanded.shape[0] * 20, 15, -1).permute(1,0,2)  # (15, BN*20, 2)
        ], dim=2)  # (15, BN20, F+2)

        space_out = self.space_encoder(t_out_features)
        space_out = space_out.permute(1, 0, 2) #B*N*20, 15, F+2
        # print("sapce ", space_out.shape)

        # print("past_rel_timed[:, -1, :].unsqueeze(1)", past_rel_timed[:, -1, :].unsqueeze(1).shape)
        # print(" space_out[ :, -10:, :].permute(0, 2, 1)",  space_out[ :, -10:, :].permute(0, 2, 1).shape)
        space_out_reshaped = space_out.view(past_rel_expanded.shape[0], 20, 15, -1) #BN, 20, 15, F+2

        best_future_scores = self.selection_net(space_out_reshaped[:,:, -10:, :].reshape(past_rel_expanded.shape[0], 20, -1) ).squeeze(-1)   # (B*N,20, 10* (F+2))--> (B*N, 20, 20)
        best_future_idx = torch.argmax(best_future_scores, dim=-1).unsqueeze(-1)  # (B*N,)

        # print("best_future_idx", best_future_idx.shape,best_future_scores.shape )
        # Gather the best trajectory
        best_future = torch.gather(
            space_out_reshaped, 1,
            best_future_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 15, space_out.shape[-1])
        ).squeeze(1)  # (B*N, 15, F+2)

        # Generate Gaussian noise
        noise = torch.randn(best_future.shape[0], 10, self.noise_dim, requires_grad=False).to(
            best_future.device)  # (B*N, 10, noise_dim)

        # print("best_future", best_future.shape, noise.shape)
        # Concatenate noise with best_future
        best_future_noisy = torch.cat([best_future[:, -10:, :], noise], dim=-1)  # (B*N, 10, F+2+noise_dim)

        # print("best_future_noisy", best_future_noisy.shape)
        # Expand mission labels to match sequence length
        # print("mission", mission.shape)
        mission = mission.view(-1, 1)  # Now shape is (64, 1)
        mission_expanded = mission.expand(-1, 8).reshape(-1).unsqueeze(1).unsqueeze(2).expand(-1, 10, 1) # (B*N, 10, 1)

        # Concatenate mission label with best_future_noisy
        final_input = torch.cat([best_future_noisy, mission_expanded], dim=-1)  # (B*N, 10, F+2+noise_dim+1)
        # Predict final future trajectory (x, y) using a simple MLP
        predicted_future = self.final_mlp(final_input)  # (B*N, 10, 2)

        # print("predicted_future", predicted_future.shape)
        # Create a mask (B*N, 10, 2), where controlled agent is zeroed out
        controlled_output = predicted_future.view(mission.shape[0],8,10,2)[:, controlled_agent, :, :]  # (B, 10, 2)
        # print("controlled_output", controlled_output.shape)
        return controlled_output


class Mission(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dropout, num_edges):
        super(Mission, self).__init__()
        self.posandspeed_to_dim = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU()
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim+2, dim_feedforward=mlp_dim, nhead=heads,
                                       dropout=dropout), num_layers=depth)  # spatial aggregation

        self.final = nn.Sequential(
            nn.Linear(8*15*(dim+4), dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(num_edges, dim),
            nn.ReLU()
        )
        self.future_encoder = nn.Sequential(
            nn.Linear(2, dim+2),
            nn.ReLU()
        )

    def forward(self, past_traj, new_traj, target, H):
        # print("mission started")
        # for name, param in self.edge_embedding.named_parameters():
        #     print(name, param.requires_grad)

        edge_feat = self.edge_embedding(H.permute(0, 2, 1))  # B, N, F
        # print("edge_feat", edge_feat.requires_grad)


        edge_node_features = torch.cat([past_traj, edge_feat.unsqueeze(2).expand(-1, -1, 5, -1)],
                                       dim=-1)  # (B, N, 5, F+2)

        edge_node_features = edge_node_features.view(edge_node_features.shape[0] * edge_node_features.shape[1],
                                                     edge_node_features.shape[2],
                                                     -1)  # Flatten time for transformer input
        # print("edge_node_features", edge_node_features.requires_grad)

        edge_node_features = edge_node_features.permute(1, 0, 2)
        # print("edge_node_features", edge_node_features.shape)
        past_rel_features = self.encoder(edge_node_features)  # learns relationships over time #B*N, 5, F+2
        past_rel_features = past_rel_features.permute(1, 0, 2)

        new_traj = new_traj.view(new_traj.shape[0] * new_traj.shape[1], 10, 2)

        future_encoder = self.future_encoder(new_traj)
        past_future_features = torch.cat([past_rel_features,future_encoder ], dim=1)  #B*N, 15, F+2
        # print("past_future_features", past_future_features.requires_grad)
        target = torch.as_tensor(target, dtype=torch.float32, device=past_future_features.device)
        target = target.unsqueeze(0).expand(past_future_features.shape[0], -1)  # (B*N, 2)
        # print("target", target.shape)
        target_expanded = target.unsqueeze(1).expand(-1 , 15, -1)  # (B*N, 15, 2)
        # print("target_expanded", target_expanded.shape)

        combined_features = torch.cat([
            target_expanded,  # (B*N, 15, 2)
            past_future_features  # (B*N, 15, F+2)
        ], dim=2) # (B*N, 15, F+4)
        # print("combined ", combined_features.shape)
        new = combined_features.view(H.shape[0], 8*combined_features.shape[1]*combined_features.shape[2])
        # print("new", new.shape)
        out = self.final(new)

        return out



class Discrimiter(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dropout, num_edges):
        super(Discrimiter, self).__init__()
        self.cat_pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )

        self.pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads,
                                       dropout=dropout), num_layers=depth)  # Time series feature extraction

        self.space_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads,
                                       dropout=dropout), num_layers=depth)  # Spatial sequence feature extraction
        self.pos_encoder = PositionalEncoding(dim)
        self.hidden_to_dim = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )


        self.future_encoder = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(num_edges, dim),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=dim + 2, nhead=heads),
            nn.Linear(dim + 2, dim),
            nn.ReLU()
        )


    def forward(self,Group_net_traj, H , past_traj,agent,agents_future_steps):

        np.random.seed(0)
        #node features past data timing
        #todo adding also features of who i see in my FOV, direction
        #Aggregation time features
        # traj_cat = torch.concat([ direction, who_I_see], dim=-1)  # B, number of agents, 5 seq, F?
        traj_cat = past_traj
        # print("traj_cat",traj_cat.shape) # B, N, 5, 2
        pos_emb = self.cat_pos_to_dim(traj_cat)  # B, N, 5, 32

        t_in = self.pos_encoder(pos_emb) # B, N, 5, 32

        # print("t_in", t_in.shape)
        t_in = t_in.view(t_in.shape[0]*t_in.shape[1],t_in.shape[2],-1)#bn, 5 , f
        t_in = t_in.permute(1, 0, 2)
        # print("t", t_in.shape)
        # print("past_traj.size(1)", past_traj.size(1), t_in.shape)
        mask = nn.Transformer().generate_square_subsequent_mask(t_in.shape[0]).to(past_traj)


        t_out = self.time_encoder(t_in, mask=mask)  # Time feature output #BN, 5, F
        t_out = t_out.permute(1,0,2)

        #group dynamics
        edge_feat = self.edge_embedding(H.permute(0, 2, 1)) #B, N, F
        edge_node_features = torch.cat([past_traj, edge_feat.unsqueeze(2).expand(-1, -1, 5, -1)], dim=-1)  # (B, N, 5, F+2)
        edge_node_features = edge_node_features.view(edge_node_features.shape[0]*edge_node_features.shape[1],edge_node_features.shape[2], -1)  # Flatten time for transformer input

        edge_node_features = edge_node_features.permute(1, 0, 2)
        # print("edge_node_features", edge_node_features.shape)
        past_rel_features = self.encoder(edge_node_features) #learns relationships over time #B*N, 5, F
        past_rel_features = past_rel_features.permute(1, 0, 2)

        # print("t_out", t_out.shape, past_rel_features.shape)
        #combine timeing with group dynamics
        past_rel_timed = t_out * past_rel_features  # (B*N, 5, F) #represents group interactions with a time-aware context

        # #future dynamics with groupnet
        # print("agents_future_steps", agents_future_steps.shape)
        # print("agents_future_steps.view",
        #       agents_future_steps.view(agents_future_steps.shape[0]*agents_future_steps.shape[1], 10, 2).shape) #BN, 10, 2
        future_encoded = self.future_encoder(agents_future_steps.view(agents_future_steps.shape[0]*agents_future_steps.shape[1], 10, 2)) #B*N, 10, F


        combined_features = torch.cat([
            past_rel_timed,  # (B*N, 5, F)
            future_encoded  # (B*N,  10, F)
        ], dim=1)

        # print("combined_features", combined_features[0])
        #Combined Past - Future(Freeze Gradients  #
        combined_features = combined_features.permute(1, 0, 2)  # (15, B*N, F)
        t_out_features = self.time_encoder(combined_features.detach()) # (15, B*N, F)

        space_out = self.space_encoder(t_out_features)
        space_out = space_out.permute(1, 0, 2) #B*N, 15, F

        score = self.final(space_out)  # (B*N, 15, 1)
        return score