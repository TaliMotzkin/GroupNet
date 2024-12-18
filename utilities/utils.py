import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
np.random.seed(42)


def reparameterized(mu, std, device):
    epsilon = Variable(torch.FloatTensor(std.size()).normal_()).to(device)
    return epsilon.mul(std).add_(mu)


def cal_ade(pred_traj, gt_traj):
    error_sum_xy = torch.sqrt(torch.sum(torch.pow(pred_traj - gt_traj, 2), dim=-1))
    error_node = torch.mean(error_sum_xy, dim=-1)
    error = torch.mean(error_node, dim=-1)
    return error


def cal_fde(pred_traj, gt_traj):
    error = torch.linalg.norm(pred_traj[:, -1] - gt_traj[:, -1], axis=-1)
    return torch.mean(error, axis=-1)


def create_hg(
    node_list: np.array,
    n_cluster: int,
    n_nearest_neighbor: int,
    n_nearest_cluster: int,
) -> list:
    hypergraph = []
    C = KMeans(n_clusters=n_cluster).fit(node_list)
    cluster_list = []
    for i in range(n_cluster):
        cluster_list.append([j for j, x in enumerate(C.labels_) if x == i])

    nearest_neighbors = NearestNeighbors(
        n_neighbors=n_nearest_neighbor, algorithm="ball_tree"
    ).fit(node_list)
    _, indices = nearest_neighbors.kneighbors(node_list)
    cluster_centers = C.cluster_centers_
    for i in range(indices.shape[0]):
        hyperedge = []

        for j in range(indices.shape[1]):
            hyperedge.append(indices[i, j])

        node_pos = node_list[i]
        dist_list = []
        for j in range(n_cluster):
            center_pos = cluster_centers[j]
            dist = math.hypot(center_pos[0] - node_pos[0], center_pos[1] - node_pos[1])
            dist_list.append(dist)
        min_value = min(dist_list)
        min_index = dist_list.index(min_value)
        for j in cluster_list[min_index]:
            hyperedge.append(j)

        hypergraph.append(hyperedge)
    return hypergraph


def softmax(input, axis=1, dim=0):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=dim)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, dim=0, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    # print(gumbel_noise)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return softmax(y / tau, dim=dim)


def gumbel_softmax(logits, tau=1, dim=0, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, dim=dim, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1) #finds the index (k) of the maximum value along the last dimension (the most probable category).
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0) #The result is a hard one-hot tensor where only one position is 1 and all others are 0.
        y = Variable(y_hard - y_soft.data) + y_soft #he y_hard tensor is discrete, but the difference (y_hard - y_soft.data) ensures that gradients flow through the soft version (y_soft).
        # print(y)
    else:
        y = y_soft
    return y


def sample_core(weights, mu): # samples a value for each batch and node from the provided Gaussian mixture model.
    original_mu_shape = mu.shape
    weights = weights.reshape(-1, weights.shape[-1])
    mu = mu.reshape(-1, mu.shape[-2], mu.shape[-1])
    categorical_distribution = torch.distributions.categorical.Categorical(weights) #categorical distribution allows sampling an index (a component) based on the given probabilities (weights).
    category = categorical_distribution.sample() #Samples an index for each row in weights based on the categorical distribution ---weights = [[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]], the sampled category might be [1, 1]
    selected_mu = torch.zeros(mu.shape[0], mu.shape[2])
    for i in range(category.shape[0]):
        selected_mu[i] = mu[i, category[i]] # Iterates over each sampled index in category and selects the corresponding mean from the flattened mu tensor
    if len(original_mu_shape) == 4:
        selected_mu = selected_mu.reshape(
            original_mu_shape[0], original_mu_shape[1], original_mu_shape[-1]
        )
    return selected_mu #contains the sampled mean values for each batch and node


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_loss(mu, target, alpha, sigma):
    original_mu_shape = mu.shape
    nll_all = torch.zeros(original_mu_shape[0], original_mu_shape[1])
    if mu.is_cuda:
        nll_all = nll_all.cuda()
    for core_index in range(mu.shape[3]):
        this_mu = mu[:, :, :, core_index, :]
        this_sigma = sigma[:, :, :, core_index, :]
        this_alpha = alpha[:, :, :, core_index]
        temp = torch.sum((this_mu - target) ** 2 / (2 * this_sigma), dim=-1)
        this_nll = this_alpha * temp
        this_nll = torch.sum(this_nll, dim=-1)
        nll_all += this_nll
    return nll_all


def compute_ade(output, target, type="sum"):
    if type == "sum":
        diff = output - target
        diff = diff**2
        diff_1 = torch.mean(torch.sqrt(torch.sum(diff, dim=3))) * 0.3
    elif type == "no_sum":
        diff = output - target
        diff = diff**2
        diff_1 = torch.mean(torch.sqrt(torch.sum(diff, dim=3)), dim=2) * 0.3
    return diff_1


def compute_fde(output, target, type="sum"):
    if type == "sum":
        diff = output - target
        diff = diff**2
        diff_2 = torch.sum(torch.sqrt(torch.sum(diff[:, :, -1, :2], dim=2))) * 0.3
    elif type == "no_sum":
        diff = output - target
        diff = diff**2
        diff_2 = torch.sqrt(torch.sum(diff[:, :, -1, :2], dim=2)) * 0.3
    return diff_2


def generate_mask(valid_list, max_num):
    mask = torch.zeros(valid_list.shape[0], max_num)
    for i in range(valid_list.shape[0]):
        mask[i, : valid_list[i]] = 1
    return mask


def custom_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def reshape_tensor(tensor: torch.Tensor, seq_length: int, nenv: int) -> torch.Tensor:
    shape = tensor.size()[1:]
    return tensor.unsqueeze(0).reshape((seq_length, nenv, *shape))


def build_dynamic_graph_and_hypergraph(z_CG, z_HG, rel_rec, rel_send, I_HG):
    """
    Builds dynamic rel_rec, rel_send, and I_HG matrices based on inferred edge/hyperedge types.

    Args:
        z_CG: Edge type probabilities, shape [B, E, L_CG].
        z_HG: Hyperedge type probabilities, shape [B, M, L_HG].
        rel_rec: Original receiver matrix [B, E, N].
        rel_send: Original sender matrix [B, E, N].
        I_HG: Original incidence matrix [B, N, M].

    Returns:
        new_rel_rec: Updated receiver matrix [B, E', N].
        new_rel_send: Updated sender matrix [B, E', N].
        new_I_HG: Updated incidence matrix [B, N, M'].
        edge_types: Edge types [B, E'].
        hyperedge_types: Hyperedge types [B, M'].
    """
    B, E, L_CG = z_CG.shape  # Batch size, number of edges, edge types
    B, M, L_HG = z_HG.shape  # Batch size, number of hyperedges, hyperedge types
    N = rel_rec.shape[-1]  # Number of nodes

    rel_send = rel_send.expand(B, -1, -1)  # Shape becomes [B, E, N]
    rel_rec = rel_rec.expand(B, -1, -1)

    # Step 1: Identify the most probable edge and hyperedge types
    edge_types = z_CG.argmax(dim=-1)  # [B, E]
    hyperedge_types = z_HG.argmax(dim=-1)  # [B, M]
    # print("edge_types", edge_types) #the argument of the edge type

    # Step 2: Filter valid edges and hyperedges (ignore "no edge" type = 0)
    valid_edges_mask = edge_types != 0  # [B, E]
    valid_hyperedges_mask = hyperedge_types != 0  # [B, M]
    # print("valid_hyperedges_mask", valid_hyperedges_mask) #true false maskes

    # print("I_HG", I_HG)
    # Initialize new graph/hypergraph matrices with zeros
    new_rel_rec = torch.zeros_like(rel_rec)  # [B, E, N]
    new_rel_send = torch.zeros_like(rel_send)  # [B, E, N]
    new_I_HG = torch.zeros_like(I_HG)  # [B, N, M]

    # Populate valid edges and hyperedges
    for b in range(B):  # Iterate over batches
        # Valid edges
        valid_edges = valid_edges_mask[b].nonzero(as_tuple=True)[0]  # Indices of valid edges
        new_rel_rec[b, valid_edges] = rel_rec[b, valid_edges]  # Preserve valid receiver nodes
        new_rel_send[b, valid_edges] = rel_send[b, valid_edges]  # Preserve valid sender nodes

        # Valid hyperedges
        valid_hyperedges = valid_hyperedges_mask[b].nonzero(as_tuple=True)[0]  # Indices of valid hyperedges
        # print(valid_hyperedges,"valid_hyperedges")
        new_I_HG[b, :, valid_hyperedges] = I_HG[b, :, valid_hyperedges]  # Preserve valid node-hyperedge relations

    return new_rel_rec, new_rel_send, new_I_HG, edge_types, hyperedge_types


def reconstruction_loss(future_traj, mu):
    """
    Computes the reconstruction loss L_Rec.

    Args:
        future_traj (torch.Tensor): Ground truth trajectories, shape [B, N, T_f, 2].
        mu (torch.Tensor): Predicted mean trajectories, shape [B, N, T_f, core_num, 2].

    Returns:
        torch.Tensor: The computed L_Rec loss.
    """
    # Step 1: Average over the core dimension to reduce mu size to [B, N, T_f, 2]
    mu_mean = mu.mean(dim=3)  # Shape: [B, N, T_f, 2]

    # Step 2: Compute the squared error
    squared_error = (future_traj - mu_mean) ** 2  # Shape: [B, N, T_f, 2]

    # Step 3: Sum over agents (N), time steps (T_f), and xy dimensions (2)
    L_Rec = squared_error.sum(dim=(1, 2, 3))  # Sum over N, T_f, and 2 for each batch

    # Step 4: Average over the batch dimension
    L_Rec = L_Rec.mean()  # Scalar loss

    return L_Rec

def compute_smoothness_loss(z_CG_list, z_HG_list, alpha_SM_CG, alpha_SM_HG):
    """
    Computes the smoothness loss between consecutive edge and hyperedge distributions.

    Args:
        z_CG_list: List of edge type distributions [B, E, L_CG].
        z_HG_list: List of hyperedge type distributions [B, M, L_HG].
        alpha_SM_CG: Smoothing coefficient for edge distributions.
        alpha_SM_HG: Smoothing coefficient for hyperedge distributions.

    Returns:
        L_SM: Smoothness loss scalar.
    """
    L_SM = 0.0
    num_graphs = len(z_CG_list)

    for t in range(num_graphs - 1):
        # Edge KL divergence
        z_CG_curr = z_CG_list[t]  # Current edge type distribution [B, E, L_CG]
        z_CG_next = z_CG_list[t + 1]  # Next edge type distribution

        kl_CG = F.kl_div(z_CG_curr.log(), z_CG_next, reduction='batchmean')  # KL divergence
        L_SM += alpha_SM_CG * kl_CG

        # Hyperedge KL divergence
        z_HG_curr = z_HG_list[t]  # Current hyperedge type distribution [B, M, L_HG]
        z_HG_next = z_HG_list[t + 1]  # Next hyperedge type distribution

        kl_HG = F.kl_div(z_HG_curr.log(), z_HG_next, reduction='batchmean')  # KL divergence
        L_SM += alpha_SM_HG * kl_HG

    return L_SM


def sharpnessLoss(z_CG_list, z_HG_list, alpha_sh_CG, alpha_sh_HG):
    """
    Computes the sharpness regularization loss L_SH over a list of Z_CG and Z_HG tensors.

    Args:
        z_CG_list (list): List of edge type distributions [B, E, L_CG].
        z_HG_list (list): List of hyperedge type distributions [B, M, L_HG].
        alpha_sh_CG (float): Weight coefficient for edge graph entropy.
        alpha_sh_HG (float): Weight coefficient for hyperedge graph entropy.

    Returns:
        torch.Tensor: Total sharpness regularization loss L_SH.
    """
    L_SH_CG = 0.0
    L_SH_HG = 0.0

    # Loop over time steps
    for z_CG, z_HG in zip(z_CG_list, z_HG_list):
        # Compute entropy for edges
        log_z_CG = torch.log(z_CG + 1e-8)  # Add small epsilon for numerical stability
        entropy_CG = -torch.sum(z_CG * log_z_CG, dim=-1)  # Sum over edge types
        L_SH_CG += torch.mean(entropy_CG)  # Average over edges and batch

        # Compute entropy for hyperedges
        log_z_HG = torch.log(z_HG + 1e-8)
        entropy_HG = -torch.sum(z_HG * log_z_HG, dim=-1)  # Sum over hyperedge types
        L_SH_HG += torch.mean(entropy_HG)  # Average over hyperedges and batch

    # Combine losses with weights
    L_SH = -alpha_sh_CG * L_SH_CG - alpha_sh_HG * L_SH_HG  # Negative because entropy is minimized

    return L_SH

def sparse_prior(num_types, device):
    q_0 = torch.zeros(num_types, device=device)
    q_0[0] = 1.0  # Probability of "no-relation" type is 1
    return q_0
def sparsity_loss(z_CG_list, z_HG_list, alpha_SP_CG, alpha_SP_HG, device):
    # Sparse prior distributions
    q_CG_0 = sparse_prior(z_CG_list[0].shape[-1], device)  # Edge types
    q_HG_0 = sparse_prior(z_HG_list[0].shape[-1], device)  # Hyperedge types
    KL_CG =0
    KL_HG = 0
    for z_CG, z_HG in zip(z_CG_list, z_HG_list):

        # Compute KL Divergence for CG
        z_CG_log = torch.log(z_CG + 1e-8)  # Avoid log(0)
        KL_CG += F.kl_div(z_CG_log, q_CG_0.expand_as(z_CG), reduction='batchmean')

        # Compute KL Divergence for HG
        z_HG_log = torch.log(z_HG + 1e-8)
        KL_HG += F.kl_div(z_HG_log, q_HG_0.expand_as(z_HG), reduction='batchmean')

        # Weighted sparsity loss
    L_SP = alpha_SP_CG * KL_CG + alpha_SP_HG * KL_HG

    return L_SP


def compute_kl_divergence_loss(z_CG_list, z_HG_list, alpha_KL_CG, alpha_KL_HG):
    """
    Computes the KL divergence loss (L_KL) between learned edge/hyperedge distributions
    and uniform priors across multiple time steps.

    Args:
        z_CG_list (list of torch.Tensor): List of Z_CG distributions over time steps.
                                         Each tensor has shape [B, E, L] (Batch, Edges, Types).
        z_HG_list (list of torch.Tensor): List of Z_HG distributions over time steps.
                                          Each tensor has shape [B, M, L] (Batch, Hyperedges, Types).
        alpha_KL_CG (float): Coefficient for the CG term.
        alpha_KL_HG (float): Coefficient for the HG term.

    Returns:
        torch.Tensor: Total KL divergence loss.
    """
    kl_loss_CG = 0.0
    kl_loss_HG = 0.0

    # Uniform priors: p = 1 / L (L = number of edge/hyperedge types)
    for z_CG in z_CG_list:
        L_CG = z_CG.shape[-1]  # Number of edge types
        uniform_prior_CG = torch.full_like(z_CG, 1 / L_CG)  # Uniform distribution
        kl_loss_CG += torch.sum(z_CG * (torch.log(z_CG + 1e-8) - torch.log(uniform_prior_CG)))

    for z_HG in z_HG_list:
        L_HG = z_HG.shape[-1]  # Number of hyperedge types
        uniform_prior_HG = torch.full_like(z_HG, 1 / L_HG)  # Uniform distribution
        kl_loss_HG += torch.sum(z_HG * (torch.log(z_HG + 1e-8) - torch.log(uniform_prior_HG)))

    # Normalize and weight the loss
    total_kl_loss = alpha_KL_CG * kl_loss_CG + alpha_KL_HG * kl_loss_HG

    return total_kl_loss