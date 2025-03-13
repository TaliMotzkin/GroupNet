import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class MLP_dict_softmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1,edge_types=5): #interaction category = edge types??
        super(MLP_dict_softmax, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(input_dim = input_dim, output_dim = self.bottleneck_dim, hidden_size=hidden_size)
        # self.dict_layer = conv1x1(self.bottleneck_dim,output_dim)
        # self.dict_layer = nn.Linear(self.bottleneck_dim,output_dim,bias=False)
        self.MLP_factor = MLP(input_dim = input_dim, output_dim = 1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim = input_dim, output_dim = input_dim, hidden_size=hidden_size)

    def forward(self, x):
        # print("MLP dict softmax x before MLP dist ", x.shape)
        x = self.init_MLP(x)
        # print("MLP dict softmax x after MLP dist ", self.MLP_distribution(x).shape)
        distribution = gumbel_softmax(self.MLP_distribution(x),tau=1/2, hard=False) #ci
        # embed = self.dict_layer(distribution)
        factor = torch.sigmoid(self.MLP_factor(x)) #strength r?
        # print("factor size ", factor.shape) #32, 400, 1
        # factor = 1
        out = factor * distribution
        # print("out size ", out.shape)
        # print("distribution size ", distribution.shape) #both 32, 400, 6
        return out, distribution

class MS_HGNN_oridinary(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, nmp_layers=4, vis=False
    ):
        super(MS_HGNN_oridinary, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.nmp_layers = nmp_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.vis = vis

        hdim_extend = 64
        self.hdim_extend = hdim_extend
        self.edge_types = 6 #todo make sure if i want to keep it 6 (5 changed) as it represents the factor number - some features related to the 400 edges?..
        self.nmp_mlp_start = MLP_dict_softmax(input_dim = hdim_extend, output_dim = h_dim, hidden_size=(128,),edge_types=self.edge_types)
        self.nmp_mlps = self.make_nmp_mlp()
        self.nmp_mlp_end = MLP(input_dim = h_dim*2, output_dim = bottleneck_dim, hidden_size=(128,))
        attention_mlp = []
        for i in range(nmp_layers):
            attention_mlp.append(MLP(input_dim=hdim_extend*2, output_dim=1, hidden_size=(32,)))
        self.attention_mlp = nn.ModuleList(attention_mlp)
        node2edge_start_mlp = []
        for i in range(nmp_layers):
            node2edge_start_mlp.append(MLP(input_dim = h_dim, output_dim = hdim_extend, hidden_size=(256,)))
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)
        edge_aggregation_list = []
        for i in range(nmp_layers):
            edge_aggregation_list.append(edge_aggregation(input_dim = h_dim, output_dim = bottleneck_dim, hidden_size=(128,),edge_types=self.edge_types))
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)

    def make_nmp_mlp(self):
        nmp_mlp = []
        for i in range(self.nmp_layers-1):
            mlp1 = MLP(input_dim = self.h_dim*2, output_dim = self.h_dim, hidden_size=(128,))
            # print("make nmp mlp1 layer, " , mlp1.shape)
            mlp2 = MLP_dict_softmax(input_dim = self.hdim_extend, output_dim = self.h_dim, hidden_size=(128,),edge_types=self.edge_types)
            # print("make nmp mlp layer after dict softmax mlp2, ", mlp2.shape)
            nmp_mlp.append(mlp1)
            nmp_mlp.append(mlp2)
        nmp_mlp = nn.ModuleList(nmp_mlp)
        return nmp_mlp

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def edge2node(self, x, rel_rec, rel_send, ori, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        H = rel_rec + rel_send
        incoming = self.edge_aggregation_list[idx](x,H,ori)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send, idx): #x= B, N, T=64
        # NOTE: Assumes that we have the same graph across all samples.
        H = rel_rec + rel_send #NXN
        x = self.node2edge_start_mlp[idx](x) #chooses batch on which to run x?
        # print("node2edge from HGNN original x ",x.shape)
        edge_init = torch.matmul(H,x) # B, 400, T=64
        # print("node2edge from HGNN original edge_init ", edge_init.shape)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:,:,None,:].transpose(2,1)).repeat(1,edge_num,1,1)
        edge_rep = edge_init[:,:,None,:].repeat(1,1,node_num,1)
        node_edge_cat = torch.cat((x_rep,edge_rep),dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:,:,:,0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight,dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight,x)
        # print("node2edge from HGNN original edge final ", edges.shape)

        return edges # B, 400, T=64

    def init_adj(self, num_ped, batch):
        off_diag = np.ones([num_ped, num_ped]) #20X20
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64) #400X20
        # print("off diag", rel_send.shape)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)


        rel_rec = rel_rec
        rel_send = rel_send
        # rel_rec = rel_rec.cuda()
        # rel_send = rel_send.cuda()

        rel_rec = rel_rec[None,:,:].repeat(batch,1,1)
        rel_send = rel_send[None,:,:].repeat(batch,1,1)

        return rel_rec, rel_send #fully communicative - creates an encoding for that..

    def forward(self, h_states):#B, N, T=64
        batch = h_states.shape[0]
        actor_num = h_states.shape[1]

        curr_hidden = h_states
        # print("HGNN forward h_states: ", curr_hidden.shape) #32, 20, 64

        # Neural Message Passing
        rel_rec, rel_send = self.init_adj(actor_num,batch) #([32, 400, 20])
        # print("HGNN forward rel_rec: ", rel_rec.shape)

        # iter 1
        edge_feat = self.node2edge(curr_hidden, rel_rec, rel_send,0) # [num_edge, h_dim*2]

        # print("HGNN edge feat: ", edge_feat.shape) #[32, 400, 64]
        # edge_feat = torch.cat([edge_feat, curr_rel_embedding], dim=2)    # [num_edge, h_dim*2+embedding_dim]
        edge_feat, factors = self.nmp_mlp_start(edge_feat)                      # [num_edge, h_dim] -> ([32, 400, 6]

        # print("edge_feat HGNN ordinal ", edge_feat.shape)
        # print("factors HGNN ordinal ", factors.shape)

        node_feat = curr_hidden

        nodetoedge_idx = 0
        if self.nmp_layers <= 1:
            pass
        else:
            for nmp_l, nmp_mlp in enumerate(self.nmp_mlps): #how many times to do edgeto node and vise verca
                if nmp_l%2==0:
                    node_feat = nmp_mlp(self.edge2node(edge_feat, rel_rec, rel_send,node_feat,nodetoedge_idx)) # [num_ped, h_dim]
                    nodetoedge_idx += 1
                else:    
                    edge_feat, _ = nmp_mlp(self.node2edge(node_feat, rel_rec, rel_send,nodetoedge_idx)) # [num_ped, h_dim] -> [num_edge, 2*h_dim] -> [num_edge, h_dim]
        node_feat = self.nmp_mlp_end(self.edge2node(edge_feat, rel_rec, rel_send, node_feat,nodetoedge_idx))
        # print("node feat final in MS_HGNN_oridinary ", node_feat.shape)#([32, 20, 64]
        # print("factors final in MS_HGNN_oridinary ", factors.shape)#> ([32, 400, 6]
        return node_feat, factors


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class MLP_dict(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1,edge_types=5):
        super(MLP_dict, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(input_dim = input_dim, output_dim = self.bottleneck_dim, hidden_size=hidden_size)
        # self.dict_layer = conv1x1(self.bottleneck_dim,output_dim)
        # self.dict_layer = nn.Linear(self.bottleneck_dim,output_dim,bias=False)
        self.MLP_factor = MLP(input_dim = input_dim, output_dim = 1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim = input_dim, output_dim = input_dim, hidden_size=hidden_size)

    def forward(self, x):
        x = self.init_MLP(x)
        distribution = torch.abs(self.MLP_distribution(x))
        return distribution, distribution

class edge_aggregation(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1, edge_types=5):
        super(edge_aggregation, self).__init__()
        self.edge_types = edge_types
        self.dict_dim = input_dim
        self.agg_mlp = []
        for i in range(edge_types):
            self.agg_mlp.append(MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=(128,)))
        self.agg_mlp = nn.ModuleList(self.agg_mlp)
        # self.embed_dict = nn.Parameter(torch.Tensor(self.edge_types, self.dict_dim))
        self.mlp = MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=(128,))

    def forward(self,edge_distribution,H,ori):
        batch = edge_distribution.shape[0]
        edges = edge_distribution.shape[1]
        edge_feature = torch.zeros(batch,edges,ori.shape[-1]).type_as(ori)
        edges = torch.matmul(H,ori)
        for i in range(self.edge_types):
            edge_feature += edge_distribution[:,:,i:i+1]*self.agg_mlp[i](edges)

        node_feature = torch.cat((torch.matmul(H.permute(0,2,1), edge_feature),ori),dim=-1)
        return node_feature

class MS_HGNN_hyper(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, nmp_layers=4, scale=2, vis=False, actor_number=11
    ): #todo changed here number of players
        super(MS_HGNN_hyper, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.nmp_layers = nmp_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.scale = scale
        self.vis = vis

        mlp_pre_dim = embedding_dim + h_dim
        self.vis = vis
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.spatial_transform = nn.Linear(h_dim,h_dim)
        hdim_extend = 64
        self.hdim_extend = hdim_extend
        self.edge_types = 10 #todo make sure we want this size 10->5

        self.nmp_mlp_start = MLP_dict_softmax(input_dim=hdim_extend, output_dim=h_dim, hidden_size=(128,),edge_types=self.edge_types)
        self.nmp_mlps = self.make_nmp_mlp()
        self.nmp_mlp_end = MLP(input_dim=h_dim*2, output_dim=bottleneck_dim, hidden_size=(128,))
        attention_mlp = []
        for i in range(nmp_layers):
            attention_mlp.append(MLP(input_dim=hdim_extend*2, output_dim=1, hidden_size=(32,)))
        self.attention_mlp = nn.ModuleList(attention_mlp)

        node2edge_start_mlp = []
        for i in range(nmp_layers):
            node2edge_start_mlp.append(MLP(input_dim = h_dim, output_dim = hdim_extend, hidden_size=(256,)))
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)
        edge_aggregation_list = []
        for i in range(nmp_layers):
            edge_aggregation_list.append(edge_aggregation(input_dim = h_dim, output_dim = bottleneck_dim, hidden_size=(128,),edge_types=self.edge_types))
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)
        self.listall = False
        if self.listall:
            if scale < actor_number:
                group_size = scale
                all_combs = []
                for i in range(actor_number): #or each actor i, generate all possible combinations of group_size - 1 other actors, excluding actor i
                    tensor_a = torch.arange(actor_number) #[0,1,2...19]
                    # tensor_a = torch.arange(actor_number).cuda()
                    tensor_a = torch.cat((tensor_a[0:i],tensor_a[i+1:]),dim=0) #all indx except of i's
                    padding = (1,0,0,0)
                    all_comb = F.pad(torch.combinations(tensor_a,r=group_size-1),padding,value=i) #generate all combinations of group sized, if 3 -> [1,2,4]....
                    all_combs.append(all_comb[None,:,:])## A tensor of shape (1, C, group_size) containing all combinations of group_size actors, including actor i, starting from number i
                self.all_combs = torch.cat(all_combs,dim=0)
                self.all_combs = self.all_combs # N, numb_comb, group size
                # self.all_combs = self.all_combs.cuda()
                # print("all_combs.shape",self.all_combs.shape)

    def make_nmp_mlp(self):
        nmp_mlp = []
        for i in range(self.nmp_layers-1):
            mlp1 = MLP(input_dim=self.h_dim*2, output_dim=self.h_dim, hidden_size=(128,))
            mlp2 = MLP_dict_softmax(input_dim=self.hdim_extend, output_dim=self.h_dim, hidden_size=(128,),edge_types=self.edge_types)
            nmp_mlp.append(mlp1)
            nmp_mlp.append(mlp2)
        nmp_mlp = nn.ModuleList(nmp_mlp)
        return nmp_mlp

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def edge2node(self, x, ori, H, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = self.edge_aggregation_list[idx](x,H,ori)
        return incoming/incoming.size(1)

    def node2edge(self, x, H, idx):
        x = self.node2edge_start_mlp[idx](x)
        edge_init = torch.matmul(H,x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:,:,None,:].transpose(2,1)).repeat(1,edge_num,1,1)
        edge_rep = edge_init[:,:,None,:].repeat(1,1,node_num,1)
        node_edge_cat = torch.cat((x_rep,edge_rep),dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:,:,:,0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight,dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight,x)
        return edges
    
    def init_adj_attention(self, feat,feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch,1,actor_number).type_as(feat)# return fully connected relationships
            return H_matrix
        group_size = scale_factor #5
        if group_size < 1:
            group_size = 1

        _,indice = torch.topk(feat_corr,dim=2,k=group_size,largest=True) #For each actor, select the top group_size neighbors based on correlation.
        #indice A tensor of shape (batch_size, actor_number, group_size) containing indices of the top correlated actors.
        H_matrix = torch.zeros(batch,actor_number,actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2,indice,1) #dim 2 scatter along the last dim - actor to actor connections, set  1 where there is a connections
        # print("indice",indice.shape) #32, 20, 5
        # print("H_matrix",H_matrix.shape) #32, 20, 20
        return H_matrix

    def init_adj_attention_listall(self, feat,feat_corr, scale_factor=2):
        batch = feat.shape[0] #32
        actor_number = feat.shape[1]#20
        if scale_factor == actor_number: #if all agents have the scale of their size so we can have inly 1 edge..
            H_matrix = torch.ones(batch,1,actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1

        #Builds an incidence matrix based on all combinations of actor correlations.
        all_indice = self.all_combs.clone() #(N,C,m) (actor_number, C, group_size = scale)
        all_indice = all_indice[None,:,:,:].repeat(batch,1,1,1) # 32, N, C, s
        all_matrix = feat_corr[:,None,None,:,:].repeat(1,actor_number,all_indice.shape[2],1,1) #added two more dims; B, N, C, N, N
        all_matrix = torch.gather(all_matrix,3,all_indice[:,:,:,:,None].repeat(1,1,1,1,actor_number)) #gather by indeces: B, N, C, s, N
        all_matrix = torch.gather(all_matrix,4,all_indice[:,:,:,None,:].repeat(1,1,1,group_size,1)) # 32, N, C, s, s
        score = torch.sum(all_matrix,dim=(3,4),keepdim=False) #32, N, C (sums)
        _,max_idx = torch.max(score,dim=2)#coses the best combination!
        indice = torch.gather(all_indice,2,max_idx[:,:,None,None].repeat(1,1,1,group_size))[:,:,0,:] #from all indeces chosing the max one from C, #B, N, s

        H_matrix = torch.zeros(batch,actor_number,actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2,indice,1) #setting values along dim 2 (the second N, like choosing a row), and will place the value 1 in the indices in ech row
        # print("H_matrix tall ", H_matrix.shape) #32, 20, 20

        return H_matrix


    def forward(self, h_states, corr):
        curr_hidden = h_states #(num_pred, h_dim) #32, N, 64, #cor = B, N, N

        if self.listall:
            H = self.init_adj_attention_listall(curr_hidden,corr,scale_factor=self.scale)
        else:
            H = self.init_adj_attention(curr_hidden,corr,scale_factor=self.scale)

        edge_hidden = self.node2edge(curr_hidden, H, idx=0)
        # print("edge_hidden ", edge_hidden.shape) #e([32, 20, 64])
        edge_feat, factor = self.nmp_mlp_start(edge_hidden)
        # print("edge_feat ", edge_feat.shape) #([32, 20, 10])
        # print("factor ", factor.shape )#([32, 20, 10])
        node_feat = curr_hidden
        node2edge_idx = 0
        if self.nmp_layers <= 1:
            pass
        else:
            for nmp_l, nmp_mlp in enumerate(self.nmp_mlps):
                if nmp_l%2==0:
                    node_feat = nmp_mlp(self.edge2node(edge_feat,node_feat,H,node2edge_idx)) 
                    node2edge_idx += 1
                else:    
                    edge_feat, _ = nmp_mlp(self.node2edge(node_feat, H, idx=node2edge_idx)) 
        node_feat = self.nmp_mlp_end(self.edge2node(edge_feat,node_feat, H,node2edge_idx))
        # print("node feat ", node_feat.shape)
        return node_feat, factor, H


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)

    # print("gumble noise , ", gumbel_noise.shape) #32,400,6
    if logits.is_cuda:
        gumbel_noise = gumbel_noise
        # gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    # print("logits for softmax ", logits.shape) #32,400,6
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    # print("y_soft after gumbe " ,y_soft.shape)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard
            # y_hard = y_hard.cuda()

        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)
