import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from config import parse_args
from torch import nn
from models import Generator, Mission, Discrimiter
import sys
from utilis_GAN import saveModel

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data.dataloader_fish import FISHDataset, seq_collate
from torch.utils.data import DataLoader
import numpy as np
import random
from model.GroupNet_nba import GroupNet
from loss import LossCompute

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def train(test_loader, args, G, M, D):
    lossfn = LossCompute(G, D, M,  args)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    optimizer_M = torch.optim.Adam(M.parameters(), lr=args.lr)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_M = torch.optim.lr_scheduler.StepLR(optimizer_M, step_size=args.lr_step, gamma=args.lr_gamma)

    iter_num = 0
    for i in range(args.epoch):
        G.train()
        for data in test_loader:
            iter_num += 1
            with torch.no_grad():
                prediction, distributions, H = model.inference(data)
            prediction = prediction.detach().clone().requires_grad_()
            H = H.detach().requires_grad_()
            prediction = prediction * args.traj_scale

            if args.method == 'mean':

                agents_future_steps = torch.mean(prediction[:, :, :10, :], dim=0)
            elif args.method == 'first':
                agents_future_steps = prediction[0, :, :10, :]

            # print("agents_future_steps", agents_future_steps.shape)
            # agents_future_steps = torch.tensor(agents_future_steps, dtype=torch.float32, device=args.device)
            final_positions = agents_future_steps.view(args.batch_size, 8, 10, 2)[:,args.agent, -1, :]  # Shape (B, 2)

            target_tensor = torch.tensor(args.target, dtype=torch.float32, device= args.device)

            distances = torch.norm(final_positions - target_tensor, dim=-1)  # Shape (B,)
            mission = (distances <= 2).float()

            optimizer_M.zero_grad()
            loss_m = lossfn.compute_Mission_loss(data["past_traj"], agents_future_steps,args.target,mission, H)

            loss_m.backward()
            optimizer_M.step()

            optimizer_D.zero_grad()
            loss_d, loss_real, loss_fake = lossfn.compute_discriminator_loss(prediction, H , data["past_traj"], mission , args.agent,  args.target, agents_future_steps)
            loss_d.backward()
            optimizer_D.step()


            optimizer_G.zero_grad()
            loss_g_all, loss_g_l2, loss_g, col_loss = lossfn.compute_generator_loss(prediction, H , data["past_traj"], mission , args.agent,  args.target, agents_future_steps)
            loss_g_all.backward()
            optimizer_G.step()

        scheduler_G.step()
        scheduler_D.step()
        scheduler_M.step()
                # print(output) #64, 10, 2
        if (i + 1) % 10 == 0:
            saveModel(G, D, M, args, str(i + 1))

def eval(test_loader, args, G):
    G.eval()
    iter_num = 0
    for i in range(args.epoch):
        for data in test_loader:
            iter_num += 1
            with torch.no_grad():
                prediction, distributions, H = model.inference(data)
            prediction = prediction.detach().clone().requires_grad_()
            H = H.detach().requires_grad_()
            prediction = prediction * args.traj_scale

            if args.method == 'mean':

                agents_future_steps = torch.mean(prediction[:, :, :10, :], dim=0)
            elif args.method == 'first':
                agents_future_steps = prediction[0, :, :10, :]

            # print("agents_future_steps", agents_future_steps.shape)
            # agents_future_steps = torch.tensor(agents_future_steps, dtype=torch.float32, device=args.device)
            final_positions = agents_future_steps.view(args.batch_size, 8, 10, 2)[:,args.agent, -1, :]  # Shape (B, 2)

            target_tensor = torch.tensor(args.target, dtype=torch.float32, device= args.device)

            distances = torch.norm(final_positions - target_tensor, dim=-1)  # Shape (B,)
            mission = (distances <= 2).float()

            pred_trajectories = G(prediction, H, data["past_traj"], mission,args.agent, args.target)


if __name__ == '__main__':
    args = parse_args()

    """ setup """
    names = [x for x in args.model_names.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device(
        'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    test_dset = FISHDataset(
        obs_len=args.past_length,
        pred_len=args.future_length,
        training=False)

    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate,
        pin_memory=True)

    for name in names:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        """ model """
        saved_path = os.path.join(args.model_save_dir, str(name) + '.p')
        print('load model from:', saved_path)
        checkpoint = torch.load(saved_path, map_location='cpu')
        training_args = checkpoint['model_cfg']

        model = GroupNet(training_args, device)
        model.set_device(device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)



        G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout, 9).to(
            args.device)

        M = Mission(args.dim, args.mlp_dim, args.depth, args.heads,args.dropout, 9)
        D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads,args.dropout, 9)
        train(test_loader, args, G, M, D)