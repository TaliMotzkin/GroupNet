import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from model.utils import initialize_weights
from model.encoder import MLPEncoder, MLP, MLP_fPIM, compute_alpha_im, MLPHGE, HyperEdgeAttention, RelationTypeInference, SeparateGRUs
from model.decoder import RNNDecoder
from utilities.utils import gumbel_softmax, build_dynamic_graph_and_hypergraph, compute_smoothness_loss, sharpnessLoss, sparsity_loss, compute_kl_divergence_loss
import math



class HGNNModelFish(nn.Module):
    def __init__(
        self,
        n_in: int, #features?
        n_head,
        n_fc_out: int, #out features
        n_hid: int, #hidden units
        M,
        Ledge,
        Lhyper,
        num_cores,
        tau: float,#temp parameter?
        hard: bool,#? for attention?
        device,
        n_layers: int = 1,
        do_prob: float = 0.0):

        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_fc_out = n_fc_out
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.do_prob = do_prob
        self.tau = tau
        self.hard = hard
        self.n_in_ec_2 = (n_hid+n_fc_out)*2
        self.n_head = n_head
        self.M = M
        self.Ledge = Ledge
        self.Lhyper = Lhyper
        self.num_cores =num_cores

        self.e_cg_2 = MLP(self.n_in_ec_2, self.n_hid, self.n_fc_out, self.do_prob)  # basic MLPs

        self.decoder = RNNDecoder(self.n_in_ec_2,self.n_fc_out,2, self.Ledge, self.Lhyper,self.n_hid, self.num_cores)

        # self.mlp2 = MLP(n_fc_out, n_fc_out, n_fc_out, do_prob)
        self.f_PIM = MLP_fPIM(self.n_hid + self.n_fc_out, self.M)
        self.encoder = MLPEncoder(self.n_head, self.n_in, self.n_hid, self.n_fc_out)
        self.f_HG_E = MLPHGE(self.n_hid + self.n_fc_out, self.n_hid , self.n_fc_out*3, do_prob)
        self.attention_hyper = HyperEdgeAttention(self.n_fc_out * 3,  self.n_hid + self.n_fc_out,  self.n_hid, self.n_fc_out * 5)
        self.gru =  SeparateGRUs(self.n_fc_out, self.Ledge, self.n_fc_out*5, self.Lhyper)

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)  # GRU mechanism? linear transformations
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)
        initialize_weights(self.hidden_r.modules()) #todo see if needed more initializations!
        initialize_weights(self.hidden_i.modules())
        initialize_weights(self.hidden_h.modules())

        self.input_r = nn.Linear(2 * n_fc_out, n_hid, bias=True)
        self.input_i = nn.Linear(2 * n_fc_out, n_hid, bias=True)
        self.input_n = nn.Linear(2 * n_fc_out, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in)



    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec,
                                 x)  # sums up the features of all nodes that receive edges, according to the graph's connectivity.
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)  # concatenates the features of the receiver and sender nodes
        return edges

    def forward(self,
            inputs,
            total_pred_steps,
            encoder_timesteps,
            recompute_gap,
            rel_rec,
            rel_send,
            tau,
            h_g = None,
            h_hg = None,
            pre_train=False):

        # print(data)
        B = inputs.size(0)
        if rel_rec.size(0) != B:
            rel_rec = rel_rec[:B].clone()  # Adjust rel_rec to match batch size
            rel_send = rel_send[:B].clone()

        v_social, v_self, alpha_ij = self.encoder(inputs, rel_rec, rel_send)
        # B, N, T, F =v_social.shape
        #print("v_social shape: ", v_social.shape)
        # v_social = v_social.view(B, N, T, -1).to(device)
        # print(v_self[0])
        #print("v_self shape: ", v_self.shape)
        v_combined = torch.cat([v_self, v_social], dim=-1)  # Shape: [B, N, 2+hidden_dim/10]
        #print("v_combined shape: ", v_combined.shape)

        #print("edges shape EC2: ", self.node2edge(v_combined, rel_rec, rel_send).shape)
        e_cg_2 = self.e_cg_2(self.node2edge(v_combined, rel_rec, rel_send))
        #print("e_cg_2 shape: ", e_cg_2.shape)

        """ The other route for the hypergraph"""
        I_PIM = self.f_PIM(v_combined)
        # print(v_combined)

        #print("I_PIM: ", I_PIM.shape)
        I_HG = gumbel_softmax(I_PIM, tau=tau, dim=-1,
                              hard=True)  # Hard sampling for binary values, what is the most probable group each i will be part of
        #print("I_HG shape:", I_HG.shape)
        # print(I_HG)

        alpha_im = compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send)
        #print("alpha_im", alpha_im.shape)

        e_HG = self.f_HG_E(alpha_im, v_combined)
        #print("e_HG", e_HG.shape)

        e_HG_2 = self.attention_hyper(e_HG, v_combined, I_HG)

        """ Getting the edges types"""
        if h_g != None:
            (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2, e_HG_2, h_g, h_hg)
        else:
            (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2,e_HG_2)

        # print("output1", output1)
        #print("edge_logits", edge_logits.shape)

        # edge_logits, hyperedge_logits =L_infer(e_cg_2,e_HG_2)
        z_CG = gumbel_softmax(edge_logits, tau=tau, dim=-1, hard=False)
        z_HG = gumbel_softmax(hyperedge_logits, tau=tau, dim=-1, hard=False)
        # print("Edge type probabilities (z_CG):", z_CG)  # [B, E, F]
        #print("Hyperedge type probabilities (z_HG):", z_HG.shape)  # [B, M, F]
        #print("z_CG", z_CG.shape)

        Z_CG_LIST = [z_CG]
        Z_HG_LIST = [z_HG]

        # todo consider printing the infered graphs
        new_rel_rec, new_rel_send, new_I_HG, new_edge_types, new_hyperedge_types = build_dynamic_graph_and_hypergraph(
            z_CG, z_HG, rel_rec, rel_send, I_HG)
        # maybe only for visualizations!!

        # print("new_rel_rec", new_rel_rec.shape)
        # print("new_rel_rec", new_rel_rec)
        # print("new_edge_types", new_edge_types)

        rel_rec = new_rel_rec
        rel_send = new_rel_send
        I_HG = new_I_HG

        """ Decoding """
        time_steps_left = total_pred_steps - encoder_timesteps - recompute_gap
        output_traj, alphas, mus, sigmas = self.decoder(inputs, z_CG, rel_rec, rel_send, z_HG, I_HG,
                                                   encoder_timesteps + recompute_gap, v_combined,
                                                   burn_in_steps=recompute_gap, pre_train = pre_train)

        output_traj = output_traj[:, :, -recompute_gap:,
                      :]  # only include the data from the last recompute_gap time steps.
        alphas = alphas[:, :, -recompute_gap:, :]
        mus = mus[:, :, -recompute_gap:, :, :]
        sigmas = sigmas[:, :, -recompute_gap:, :, :]

        if recompute_gap < encoder_timesteps:
            inputs = torch.cat(
                (inputs[:, :, -(encoder_timesteps - recompute_gap):, :], output_traj),
                dim=2,
            )  # last part of the inputs that wasn't covered by the new outputs
        else:
            inputs = output_traj[:, :, -encoder_timesteps:, :]  # completely refreshing it,

        output_lists = {
            "z_CG": [z_CG],
            "z_HG": [z_HG],
            "trajs": [output_traj],
            "alphas": [alphas],
            "mus": [mus],
            "sigmas": [sigmas],
        }

        num_new_graph = math.ceil((total_pred_steps - encoder_timesteps) / recompute_gap) - 1
        #print(num_new_graph)
        for _ in range(num_new_graph):
            # print(data)
            v_social, v_self, alpha_ij = self.encoder(inputs, rel_rec, rel_send)
            # B, N, T, F =v_social.shape
            # print("v_social shape: ", v_social.shape)
            # v_social = v_social.view(B, N, T, -1).to(device)
            # print(v_self[0])
            # print("v_self shape: ", v_self.shape)
            v_combined = torch.cat([v_self, v_social], dim=-1)  # Shape: [B, N, 2+hidden_dim/10]
            # print("v_combined shape: ", v_combined.shape)

            # print("edges shape EC2: ", self.node2edge(v_combined, rel_rec, rel_send).shape)
            e_cg_2 =  self.e_cg_2(self.node2edge(v_combined, rel_rec, rel_send))
            # print("e_cg_2 shape: ", e_cg_2.shape)

            """ The other route for the hypergraph"""
            I_PIM = self.f_PIM(v_combined)
            # print(v_combined)

            # print("I_PIM: ", I_PIM.shape)
            I_HG = gumbel_softmax(I_PIM, tau=tau, dim=-1,
                                  hard=True)  # Hard sampling for binary values, what is the most probable group each i will be part of
            # print("I_HG shape:", I_HG.shape)
            # print(I_HG)

            alpha_im = compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send)
            #print("alpha_im", alpha_im.shape)

            e_HG = self.f_HG_E(alpha_im, v_combined)
            #print("e_HG", e_HG.shape)

            e_HG_2 = self.attention_hyper(e_HG, v_combined, I_HG)

            """ Getting the edges types"""
            if h_g != None:
                (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2, e_HG_2, h_g, h_hg)
            else:
                (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2,e_HG_2)

            # print("output1", output1)
            #print("edge_logits", edge_logits.shape)

            # edge_logits, hyperedge_logits =L_infer(e_cg_2,e_HG_2)
            z_CG = gumbel_softmax(edge_logits, tau=tau, dim=-1, hard=False)
            z_HG = gumbel_softmax(hyperedge_logits, tau=tau, dim=-1, hard=False)
            # print("Edge type probabilities (z_CG):", z_CG)  # [B, E, F]
            #print("Hyperedge type probabilities (z_HG):", z_HG.shape)  # [B, M, F]
            #print("z_CG", z_CG.shape)

            Z_CG_LIST.append(z_CG)
            Z_HG_LIST.append(z_HG)

            # todo consider printing the infered graphs
            new_rel_rec, new_rel_send, new_I_HG, new_edge_types, new_hyperedge_types = build_dynamic_graph_and_hypergraph(
                z_CG, z_HG, rel_rec, rel_send, I_HG)
            # maybe only for visualizations!!

            # print("new_rel_rec", new_rel_rec.shape)
            # print("new_rel_rec", new_rel_rec)
            # print("new_hyperedge_types", new_hyperedge_types)
            # print("new_I_HG", new_I_HG)

            rel_rec = new_rel_rec
            rel_send = new_rel_send
            I_HG = new_I_HG

            """ Decoding """
            output_traj, alphas, mus, sigmas = self.decoder(inputs, z_CG, rel_rec, rel_send, z_HG, I_HG,
                                                       encoder_timesteps + recompute_gap, v_combined,
                                                       burn_in_steps=recompute_gap)

            if time_steps_left >= recompute_gap:  # he length of the output trajectory and associated parameters aligns
                output_traj = output_traj[:, :, -recompute_gap:, :]
                alphas = alphas[:, :, -recompute_gap:, :]
                mus = mus[:, :, -recompute_gap:, :, :]
                sigmas = sigmas[:, :, -recompute_gap:, :, :]
            else:
                output_traj = output_traj[:, :, -time_steps_left:, :]
                alphas = alphas[:, :, -time_steps_left:, :]
                mus = mus[:, :, -time_steps_left:, :, :]
                sigmas = sigmas[:, :, -time_steps_left:, :, :]

            time_steps_left -= recompute_gap
            if recompute_gap < encoder_timesteps:  # integrating new trajectories while considering the overlap with previous encoder timestep
                inputs = torch.cat(
                    (inputs[:, :, -(encoder_timesteps - recompute_gap):, :], output_traj),
                    dim=2,
                )
            else:
                inputs = output_traj[:, :, -encoder_timesteps:, :]

            output_lists["z_CG"].append(z_CG)
            output_lists["z_HG"].append(z_HG)
            output_lists["trajs"].append(output_traj)
            output_lists["alphas"].append(alphas)
            output_lists["mus"].append(mus)
            output_lists["sigmas"].append(sigmas)

        L_SM = compute_smoothness_loss(Z_CG_LIST,Z_HG_LIST,0.1,0.1)
        L_SH = sharpnessLoss(Z_CG_LIST,Z_HG_LIST,0.1,0.1)
        L_SP = sparsity_loss(Z_CG_LIST,Z_HG_LIST,0.1,0.1, self.device)
        L_KL = compute_kl_divergence_loss(Z_CG_LIST,Z_HG_LIST,0.1,0.1)

        return output_lists,h_g, h_hg, rel_rec, rel_send, L_SM, L_SH, L_SP, L_KL

    def inference(self, inputs,
            total_pred_steps,
            encoder_timesteps,
            recompute_gap,
            rel_rec,
            rel_send,
            tau,
            h_g = None,
            h_hg = None,
            pre_train=False):

        inputs = inputs['past_traj']
        B = inputs.size(0)
        if rel_rec.size(0) != B:
            rel_rec = rel_rec[:B].clone()  # Adjust rel_rec to match batch size
            rel_send = rel_send[:B].clone()

        v_social, v_self, alpha_ij = self.encoder(inputs, rel_rec, rel_send)
        v_combined = torch.cat([v_self, v_social], dim=-1)  # Shape: [B, N, 2+hidden_dim/10]
        e_cg_2 = self.e_cg_2(self.node2edge(v_combined, rel_rec, rel_send))

        """ The other route for the hypergraph"""
        I_PIM = self.f_PIM(v_combined)
        I_HG = gumbel_softmax(I_PIM, tau=tau, dim=-1,
                              hard=True)
        alpha_im = compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send)
        e_HG = self.f_HG_E(alpha_im, v_combined)
        e_HG_2 = self.attention_hyper(e_HG, v_combined, I_HG)

        """ Getting the edges types"""
        if h_g != None:
            (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2, e_HG_2, h_g, h_hg)
        else:
            (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2,e_HG_2)

        z_CG = gumbel_softmax(edge_logits, tau=tau, dim=-1, hard=False)
        z_HG = gumbel_softmax(hyperedge_logits, tau=tau, dim=-1, hard=False)

        # todo consider printing the infered graphs
        new_rel_rec, new_rel_send, new_I_HG, new_edge_types, new_hyperedge_types = build_dynamic_graph_and_hypergraph(
            z_CG, z_HG, rel_rec, rel_send, I_HG)
        # maybe only for visualizations!!

        # print("new_rel_rec", new_rel_rec.shape)
        # print("new_rel_rec", new_rel_rec)
        # print("new_edge_types", new_edge_types)

        rel_rec = new_rel_rec
        rel_send = new_rel_send
        I_HG = new_I_HG

        """ Decoding """
        time_steps_left = total_pred_steps - encoder_timesteps - recompute_gap
        output_traj, alphas, mus, sigmas = self.decoder(inputs, z_CG, rel_rec, rel_send, z_HG, I_HG,
                                                   encoder_timesteps + recompute_gap, v_combined,
                                                   burn_in_steps=recompute_gap, pre_train = pre_train)

        output_traj = output_traj[:, :, -recompute_gap:,
                      :]  # only include the data from the last recompute_gap time steps.


        if recompute_gap < encoder_timesteps:
            inputs = torch.cat(
                (inputs[:, :, -(encoder_timesteps - recompute_gap):, :], output_traj),
                dim=2,
            )  # last part of the inputs that wasn't covered by the new outputs
        else:
            inputs = output_traj[:, :, -encoder_timesteps:, :]  # completely refreshing it,

        output_lists = {
            "trajs": [output_traj],
        }

        num_new_graph = math.ceil((total_pred_steps - encoder_timesteps) / recompute_gap) - 1

        for _ in range(num_new_graph):
            v_social, v_self, alpha_ij = self.encoder(inputs, rel_rec, rel_send)
            v_combined = torch.cat([v_self, v_social], dim=-1)  # Shape: [B, N, 2+hidden_dim/10]
            e_cg_2 =  self.e_cg_2(self.node2edge(v_combined, rel_rec, rel_send))

            """ The other route for the hypergraph"""
            I_PIM = self.f_PIM(v_combined)

            I_HG = gumbel_softmax(I_PIM, tau=tau, dim=-1,
                                  hard=True)  # Hard sampling for binary values, what is the most probable group each i will be part of


            alpha_im = compute_alpha_im(alpha_ij, I_HG, rel_rec, rel_send)


            e_HG = self.f_HG_E(alpha_im, v_combined)

            e_HG_2 = self.attention_hyper(e_HG, v_combined, I_HG)

            """ Getting the edges types"""
            if h_g != None:
                (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2, e_HG_2, h_g, h_hg)
            else:
                (edge_logits, h_g), (hyperedge_logits, h_hg) = self.gru(e_cg_2,e_HG_2)


            z_CG = gumbel_softmax(edge_logits, tau=tau, dim=-1, hard=False)
            z_HG = gumbel_softmax(hyperedge_logits, tau=tau, dim=-1, hard=False)


            # todo consider printing the infered graphs
            new_rel_rec, new_rel_send, new_I_HG, new_edge_types, new_hyperedge_types = build_dynamic_graph_and_hypergraph(
                z_CG, z_HG, rel_rec, rel_send, I_HG)
            # maybe only for visualizations!!

            # print("new_rel_rec", new_rel_rec.shape)
            # print("new_rel_rec", new_rel_rec)
            # print("new_edge_types", new_edge_types)

            rel_rec = new_rel_rec
            rel_send = new_rel_send
            I_HG = new_I_HG

            """ Decoding """
            output_traj, alphas, mus, sigmas = self.decoder(inputs, z_CG, rel_rec, rel_send, z_HG, I_HG,
                                                       encoder_timesteps + recompute_gap, v_combined,
                                                       burn_in_steps=recompute_gap)

            if time_steps_left >= recompute_gap:  # he length of the output trajectory and associated parameters aligns
                output_traj = output_traj[:, :, -recompute_gap:, :]

            else:
                output_traj = output_traj[:, :, -time_steps_left:, :]


            time_steps_left -= recompute_gap
            if recompute_gap < encoder_timesteps:  # integrating new trajectories while considering the overlap with previous encoder timestep
                inputs = torch.cat(
                    (inputs[:, :, -(encoder_timesteps - recompute_gap):, :], output_traj),
                    dim=2,
                )
            else:
                inputs = output_traj[:, :, -encoder_timesteps:, :]


            output_lists["trajs"].append(output_traj)


        return output_lists


