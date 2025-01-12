import math
import torch
import random
import itertools
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utilities.utils import gumbel_softmax, custom_softmax, sample_core

class MLP(nn.Module):
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        print(f'MLP forward :{inputs.shape}')
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.fc3(x)
        return x

class RNNDecoder(nn.Module):
    def __init__(
        self,
        n_in_mlp,
        n_out,
        n_in_node,
        edge_types, #number?
        edge_types_hg,
        n_hid,
        num_cores=1, #for attention?
        do_prob=0.0,
        skip_first=False,
        env_flag=False,
        n_env_in=0,
    ):
        super(RNNDecoder, self).__init__()
        self.num_cores = num_cores
        self.n_in_mlp = n_in_mlp
        self.n_hid = n_hid
        self.n_out = n_out
        self.dim = n_in_node
        self.env_flag = env_flag
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first
        self.in_hyper = int(n_in_mlp/2)
        # print('n_in_mlp', n_in_mlp)
        self.shortcut = nn.Linear(self.n_in_mlp, self.msg_out_shape) if self.n_in_mlp != self.msg_out_shape else nn.Identity()

        self.msg_fc1_g =nn.Sequential(nn.Linear(n_in_mlp, n_hid), nn.BatchNorm1d(n_hid))
        self.msg_fc1_hg = nn.Sequential(nn.Linear(self.in_hyper, n_hid), nn.BatchNorm1d(n_hid))
        self.msg_fc2_g = nn.Sequential( nn.Linear(n_hid, n_hid),  nn.BatchNorm1d(n_hid))
        self.msg_fc2_hg = nn.Sequential( nn.Linear(n_hid, n_hid),  nn.BatchNorm1d(n_hid))

        self.f_CG_e_l = MLP(self.n_in_mlp, self.n_hid, self.n_out)
        self.relu = nn.LeakyReLU(negative_slope=0.01)


        self.out_fc1 = nn.Linear(n_hid * 2, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.W_alpha = nn.Linear(n_hid, self.num_cores)
        self.W_mu = nn.Linear(n_hid, self.num_cores * n_in_node)
        self.W_sigma = nn.Linear(n_hid, self.num_cores * n_in_node)

        self.dropout_prob = do_prob

    def single_step_forward(
        self,
        inputs,
        rel_rec_g,
        rel_send_g,
        Z_CG,
        I_HG,
        Z_HG,
        v_combined,
        hidden_hg,

        pre_train,
    ):
        receivers = torch.matmul(rel_rec_g, v_combined) #this is same as nodeto edge
        senders = torch.matmul(rel_send_g, v_combined)
        pre_msg = torch.cat([receivers, senders], dim=-1) #preliminary messages #B, E, 2*H (H=n_hid+n_out)
        # print("pre_msg", pre_msg.shape)
        B, E, H = pre_msg.shape
        all_msgs = torch.zeros(pre_msg.size()[0], pre_msg.size()[1], self.n_hid)#tensor all_msgs to accumulate messages after processing
        # print("all_msgs1", all_msgs.shape)

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2_g)) - 1.0 #layers in ModuleList which defined by the edge type?
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2_g))

        # print(i)
        shortcut = self.shortcut(pre_msg)
        # print("shortcut", shortcut.shape)
        new_pre = pre_msg.flatten(start_dim=0, end_dim=1)
        # print("pre_msg", new_pre.shape)
        msg = self.relu(self.msg_fc1_g(new_pre))
        msg = F.dropout(msg, p=self.dropout_prob)
        msg = self.relu(self.msg_fc2_g(msg))
        msg = msg.view(B, E, self.n_hid)
        # print("msg", msg.shape)
        Z_CG = Z_CG.unsqueeze(-1).expand(-1, -1, -1, self.n_hid).permute(0, 1, 3, 2)
        all_msgs = self.relu((msg.unsqueeze(-1) * Z_CG).mean(dim=-1)  + shortcut) #the  first dimension could represent batches of data.

        # all_msgs = msg * Z_CG[:, :, i : i + 1] #the  first dimension could represent batches of data.

        # The second dimension - different nodes or edges. The third dimension represents different edge types?
            # all_msgs += msg / norm #all masseges are e_CG_ij
        # print("all_msgs", all_msgs.shape, rel_send_g.shape)
        # print("msg", msg.shape)

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_send_g).transpose(-2, -1) #aggregates incoming messages to each node?
        # print("agg_msgs after agg", agg_msgs.shape)
        hidden_g = agg_msgs.contiguous() / inputs.size(2) #~e_CG_ij
        # print("agg_msgs after agg norm", hidden_g.shape)


        if pre_train:
            hidden_hg = torch.zeros(hidden_g.size()) #same size HG if pretrained- not making new prior knowledge
            if inputs.is_cuda:
                hidden_hg = hidden_hg.cuda()
        if not pre_train: #same as for the G!
            pre_msg = torch.einsum('bnm,bnf->bmf', I_HG, v_combined)  # Aggregate nodes to hyperedges
            # print("pre_msg hg" , pre_msg.shape)
            B_hg , E_hg, F_hg = pre_msg.shape

            all_msgs =  torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
            if inputs.is_cuda:
                all_msgs = all_msgs.cuda()

            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2_hg)) - 1.0
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2_hg))


            # print("pre_msg", pre_msg.shape)
            new_pre = pre_msg.flatten(start_dim=0, end_dim=1)
            # print(new_pre.shape, "new_pre")
            msg = self.relu(self.msg_fc1_hg(new_pre))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = self.relu(self.msg_fc2_hg(msg))
            msg = msg.view(B_hg , E_hg,  self.n_hid)
            # print("msg", msg.shape)
            Z_HG = Z_HG.unsqueeze(-1).expand(-1, -1, -1, self.n_hid).permute(0, 1, 3, 2)
            all_msgs = (msg.unsqueeze(-1) * Z_HG).mean(dim=-1)

                # print("all_msgs loop", all_msgs.shape)
                # print("msg", msg.shape)

            #print("all_msgs hg1", all_msgs.shape, I_HG.shape)
            agg_msgs = all_msgs.transpose(-2, -1).matmul(I_HG.transpose(-2, -1)).transpose(-2, -1)
            #print("agg_msgs hg", agg_msgs.shape)
            hidden_hg = agg_msgs.contiguous() / inputs.size(2)
            #print("hidden_hg hg", hidden_hg.shape)



        v = F.dropout(
            self.relu(self.out_fc1(torch.cat((hidden_g, hidden_hg), dim=-1))),
            p=self.dropout_prob,)

        # v = F.dropout(F.relu(self.out_fc2(v)), p=self.dropout_prob)

        #print("v", v.shape)
        # print(pred[0,0])

        alpha = self.W_alpha(v) #alpha represents mixing weights (computed using softmax for normalization), and mu and sigma are the means and variances
        alpha = F.softmax(alpha, dim=-1)
        #print("alpha", alpha.shape)
        mu = self.W_mu(v)
        mu = mu.reshape(mu.shape[0], mu.shape[1], self.num_cores, self.dim)
        sigma = torch.ones((mu.shape[0], mu.shape[1], self.num_cores, self.dim)) * 1
        #print("sigma", sigma.shape)
        if inputs.is_cuda:
            sigma = sigma.cuda()

        pred = sample_core(alpha, mu)
        #print("after sample core?", pred.shape)
        for i in range(mu.shape[2]):
            mu[:, :, i, :] += inputs #offset the transformation done by the model??
        if inputs.is_cuda:
            pred = pred.cuda()
        #print("last mu", mu.shape)
        noise = torch.randn_like(pred) * sigma.mean(dim=2)  # todo - not sure if adding - Use sigma to scale the noise
        pred = pred + noise

        pred = inputs + pred
        #print("lsst pred ", pred.shape)
        return pred, alpha, mu, sigma, hidden_g, hidden_hg
        # return 1, 1, 1, 1, 1, 1

    def forward(
        self,
        data,
        rel_type_g,
        rel_rec_g,
        rel_send_g,
        rel_type_hg,
        I_HG,
        output_steps,
        v_combined,
        burn_in_steps=1,
        pre_train=False,
    ):

        #data == B, N ,T, F
        inputs = data.transpose(1, 2).contiguous() #switch the feature and time dimensions?
        # data == B, T,N, F
        time_steps = inputs.size(1)


        pred_all = []
        alpha_all = []
        mu_all = []
        sigma_all = []
        hidden_hg =1

        for step in range(output_steps):
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ step", step)
            if step < burn_in_steps:
                ins = inputs[:, step, :, :] #takes only 1 T?...
            else:
                ins = pred_all[step - 1]

            #print("ins", ins.shape)
            pred, alpha, mu, sigma, hidden_g, hidden_hg = self.single_step_forward(
                ins,
                rel_rec_g,
                rel_send_g,
                rel_type_g,
                I_HG,
                rel_type_hg,
                v_combined,
                hidden_hg,
                pre_train,
            )
            pred_all.append(pred)
            mu_all.append(mu)
            alpha_all.append(alpha)
            sigma_all.append(sigma)

        preds = torch.stack(pred_all, dim=1)
        alphas = torch.stack(alpha_all, dim=2)
        mus = torch.stack(mu_all, dim=2)
        sigmas = torch.stack(sigma_all, dim=2)

        # print(preds.transpose(1, 2).contiguous().shape, alphas.shape, mus.shape, sigmas.shape)
        # print(preds.transpose(1, 2).contiguous())
        # print(data)
        return preds.transpose(1, 2).contiguous(), alphas, mus, sigmas