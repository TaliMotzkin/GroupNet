import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from model.utils import initialize_weights
from model.encoder import MLPEncoder, MLP





class HGNNModelFish(nn.Module):
    def __init__(
        self,
        n_in: int, #features?
        n_fc_out: int, #out features
        n_hid: int, #hidden units
        tau: float,#temp parameter?
        hard: bool,#? for attention?
        device,
        n_layers: int = 1,
        do_prob: float = 0.0):

        super(HGNNModelFish).__init__()
        self.device = device
        self.n_in = n_in
        self.n_fc_out = n_fc_out
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.do_prob = do_prob
        self.tau = tau
        self.hard = hard
        self.n_in_ec_2 = (n_hid+n_fc_out)*2

        self.encoder = MLPEncoder(self.n_in, self.n_hid, self.n_fc_out, "GAT")
        # self.decoder = Decoder()
        self.e_cg_2 = MLP(self.n_in_ec_2,  self.n_hid,  self.n_hid, do_prob)  # basic MLPs
        # self.mlp2 = MLP(n_fc_out, n_fc_out, n_fc_out, do_prob)


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

        def forward(
                self,
                x: torch.Tensor = None, #previous trajectories for the developing graph process
                inputs=None,
                total_pred_steps=None,  # the number of prediction steps the network should output
                encoder_timesteps=None,  # number of timesteps that the encoder processes
                recompute_gap=None,  # how often certain computations are refreshed or recalculated?
                var=None,  #todo not sure if needed and for what
                agent_types=None,
                pre_train: bool = False,
        ):
            if x is not None:
                return self._forward_trajectory(x)
            else:
                return self._forward_graph(
                    inputs,
                    total_pred_steps,
                    encoder_timesteps,
                    recompute_gap,
                    var,
                    agent_types,
                    pre_train,
                )

        def node2edge(self, x, rel_rec, rel_send):
            receivers = torch.matmul(rel_rec,
                                     x)  # sums up the features of all nodes that receive edges, according to the graph's connectivity.
            senders = torch.matmul(rel_send, x)
            edges = torch.cat([receivers, senders], dim=2)  # concatenates the features of the receiver and sender nodes
            return edges

        def _forward_graph(  # for the hyper graph
                self,
                inputs,
                total_pred_steps,
                encoder_timesteps,
                recompute_gap,
                var,
                agent_types,
                pre_train=False,
        ):
            v_social, v_self = MLPEncoder(inputs, rel_rec1, rel_send1)
            # B, N, T, F =v_social.shape
            print("v_social shape: ", v_social.shape)
            # v_social = v_social.view(B, N, T, -1).to(device)
            print("v_self shape: ", v_self.shape)
            v_combined = torch.cat([v_self, v_social], dim=-1)  # Shape: [B, N, 2+hidden_dim/10]
            print("v_combined shape: ", v_combined.shape)

            print("edges shape EC2: ", node2edge(v_combined, rel_rec1, rel_send1).shape)
            e_cg_2 = self.e_cg_2(node2edge(v_combined, rel_rec1, rel_send1))
            print("e_cg_2 shape: ", e_cg_2.shape)


            hidden = torch.zeros((graph.size(0), graph.size(1), self.n_hid))
            if inputs.is_cuda:
                hidden = hidden.cuda()
            r = torch.sigmoid(self.input_r(x1))  # GRU?
            i = torch.sigmoid(self.input_i(x1))
            n = torch.tanh(self.input_n(x1))
            hidden = (1 - i) * n + i * hidden

            output_g_graph = F.dropout(F.relu(self.out_fc1(hidden)), p=self.do_prob)
            output_g_graph = F.dropout(F.relu(self.out_fc2(output_g_graph)), p=self.do_prob)
            output_g_graph = self.out_fc3(output_g_graph)
            output_g_prob = custom_softmax(output_g_graph, -1)
            output_g_graph = gumbel_softmax(output_g_graph, tau=self.tau, hard=self.hard)

            output_hg_graph = None
            if not pre_train:  # same for hypergraph?
                h_graph = encoder2.forward(inputs, rel_rec2, rel_send2)  # node to edge
                h_graph[:, :, :] = h_graph[0, 0]
                x2 = self.mlp1(h_graph)
                x2 = self.edge2node(x2, rel_rec2, rel_send2)
                x2 = self.mlp2(x2)
                x2 = self.node2edge(x2, rel_rec2, rel_send2)

                h_hidden = torch.zeros((h_graph.size(0), h_graph.size(1), self.n_hid))
                if inputs.is_cuda:
                    h_hidden = h_hidden.cuda()
                r = torch.sigmoid(self.input_r(x2))
                i = torch.sigmoid(self.input_i(x2))
                n = torch.tanh(self.input_n(x2))
                h_hidden = (1 - i) * n + i * h_hidden

                output_hg_graph = F.dropout(F.relu(self.out_fc1(h_hidden)), p=self.do_prob)
                output_hg_graph = F.dropout(F.relu(self.out_fc2(output_hg_graph)), p=self.do_prob)
                output_hg_graph = self.out_fc3(output_hg_graph)
                output_hg_graph = gumbel_softmax(output_hg_graph, tau=self.tau, hard=self.hard)

            time_steps_left = total_pred_steps - encoder_timesteps - recompute_gap  # hoe much time left for decoding
            output_traj, alphas, mus, sigmas = decoder.forward(
                inputs,
                output_g_graph,
                rel_rec1,
                rel_send1,
                output_hg_graph,
                rel_rec2,
                rel_send2,  # besed on what we decide this matrixes?
                encoder_timesteps + recompute_gap,
                var,
                burn_in=True,  # flags
                burn_in_steps=encoder_timesteps,
                pre_train=pre_train,
            )

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
                "g_graphs": [output_g_graph],
                "hg_graphs": [output_hg_graph],
                "probs": [output_g_prob],  # no such thing
                "trajs": [output_traj],
                "alphas": [alphas],
                "mus": [mus],
                "sigmas": [sigmas],
            }

            num_new_graph = math.ceil(
                (total_pred_steps - encoder_timesteps) / recompute_gap
            ) - 1  # determines how many new graph iterations are needed

            for _ in range(num_new_graph):  # a new time window or sequence of graph processing?
                graph = encoder1.forward(inputs, rel_rec1, rel_send1)
                graph[:, :, :] = graph[0, 0]
                x1 = self.mlp1(graph)
                x1 = self.edge2node(x1, rel_rec1, rel_send1)
                x1 = self.mlp2(x1)
                x1 = self.node2edge(x1, rel_rec1, rel_send1)

                r = torch.sigmoid(self.input_r(x1))
                i = torch.sigmoid(self.input_i(x1))
                n = torch.tanh(self.input_n(x1))
                hidden = (1 - i) * n + i * hidden

                output_g_graph = F.dropout(F.relu(self.out_fc1(hidden)), p=self.do_prob)
                output_g_graph = F.dropout(F.relu(self.out_fc2(output_g_graph)), p=self.do_prob)
                output_g_graph = self.out_fc3(output_g_graph)
                output_g_prob = custom_softmax(output_g_graph, -1)
                output_g_graph = gumbel_softmax(output_g_graph, tau=self.tau, hard=self.hard)

                output_hg_graph = None
                if not pre_train:
                    h_graph = encoder2.forward(inputs, rel_rec2, rel_send2)
                    h_graph[:, :, :] = h_graph[0, 0]
                    x2 = self.mlp1(h_graph)
                    x2 = self.edge2node(x2, rel_rec2, rel_send2)
                    x2 = self.mlp2(x2)
                    x2 = self.node2edge(x2, rel_rec2, rel_send2)

                    r = torch.sigmoid(self.input_r(x2))
                    i = torch.sigmoid(self.input_i(x2))
                    n = torch.tanh(self.input_n(x2))
                    h_hidden = (1 - i) * n + i * h_hidden

                    output_hg_graph = F.dropout(F.relu(self.out_fc1(h_hidden)), p=self.do_prob)
                    output_hg_graph = F.dropout(F.relu(self.out_fc2(output_hg_graph)), p=self.do_prob)
                    output_hg_graph = self.out_fc3(output_hg_graph)
                    output_hg_graph = gumbel_softmax(output_hg_graph, tau=self.tau, hard=self.hard)

                output_traj, alphas, mus, sigmas = decoder.forward(
                    inputs,
                    output_g_graph,
                    rel_rec1,
                    rel_send1,
                    output_hg_graph,
                    rel_rec2,
                    rel_send2,
                    encoder_timesteps + recompute_gap,
                    var,
                    burn_in=True,
                    burn_in_steps=encoder_timesteps,
                    pre_train=pre_train,
                )

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

                output_lists["g_graphs"].append(output_g_graph)
                output_lists["hg_graphs"].append(output_hg_graph)
                output_lists["probs"].append(output_g_prob)
                output_lists["trajs"].append(output_traj)
                output_lists["alphas"].append(alphas)
                output_lists["mus"].append(mus)
                output_lists["sigmas"].append(sigmas)

            return (
                output_lists["trajs"],
                output_lists["g_graphs"],
                output_lists["hg_graphs"],
                output_lists["probs"],
                output_lists["alphas"],
                output_lists["mus"],
                output_lists["sigmas"],
            )
