import os
import sys
import argparse
import time
import numpy as np
import torch
import random
from torch import optim
from torch.optim import lr_scheduler

sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from data.dataloader_fish import FISHDataset, seq_collate
from data.dataloader_nba import NBADataset
from model.GroupNet_nba import GroupNet
import math
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='nba')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--past_length', type=int, default=5)
parser.add_argument('--future_length', type=int, default=10)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--learn_prior', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)
parser.add_argument('--iternum_print', type=int, default=50)

parser.add_argument('--ztype', default='gaussian')
parser.add_argument('--zdim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--hyper_scales', nargs='+', type=int,
                    default=[5, 8])  # todo make sure 5,11 unrelated to 11 players?
parser.add_argument('--num_decompose', type=int, default=2)
parser.add_argument('--min_clip', type=float, default=2.0)

parser.add_argument('--model_save_dir', default='saved_models/fish_overlap')
parser.add_argument('--model_save_epoch', type=int, default=2)

parser.add_argument('--epoch_continue', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

""" setup """
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
print('device:', device)
print(args)


def validate(validation_loader):
    model.eval()
    total_val_loss = 0
    iter = 0
    with torch.no_grad():
        for data in validation_loader:
            total_loss, loss_pred, loss_recover, loss_kl, loss_diverse, my_loss, _, _, _, _ = model(data)
            total_val_loss += total_loss.item()
            iter += 1

    avg_val_loss = total_val_loss / iter
    print(f'total avg validation Loss: {avg_val_loss:.3f}')
    print("other val losses: loss pred", loss_pred, "loss recover ", loss_recover, "loss kl ", loss_kl, "loss diverse ",
          loss_diverse)
    return avg_val_loss

def plot_variance(T, N, variance_per_agent):
    time_steps = np.arange(T)
    width = 0.08

    fig, ax = plt.subplots(figsize=(12, 6))

    for agent in range(N):
        x_pos = time_steps + (agent - N / 2) * (2 * width / N)  #bars for each agent
        ax.bar(x_pos, variance_per_agent[agent, :, 0], width=width / N, label=f'Agent {agent} X')
        ax.bar(x_pos + width / 2, variance_per_agent[agent, :, 1], width=width / N, label=f'Agent {agent} Y')


    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Variance")
    ax.set_title("Variance per Agent for X and Y Coordinates Over Time")
    ax.legend(ncol=4, fontsize=8)
    plt.xticks(time_steps)
    plt.show()
    plt.savefig("sample_plot.png")
    plt.close()


def train(train_loader, epoch, batch_size, device):
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    avg_loss = 0
    all_loss = 0
    trajectory_wins = {agent: np.zeros(20, dtype=int) for agent in range(8)}
    for data in train_loader:
        total_loss, loss_pred, loss_recover, loss_kl, loss_diverse, my_loss, outputs_logits, predictions, actual_probs, true_indices = model(
            data)

        """ optimize """
        optimizer.zero_grad()
        total_loss.backward()
        if iter_num % args.iternum_print == 0:
            print(f"outputs_logits: {outputs_logits[0].detach().cpu().numpy()}")
            # print("predictions", predictions[0, :3])
            print("actual_probs", actual_probs[:2])
            for name, param in model.named_parameters():
                if name.startswith("final_model") and param.grad is not None:
                    print(f"{name} gradient: {param.grad.abs().mean().item()}")


        optimizer.step()

        B_N, S, T, C = predictions.shape
        N = B_N// batch_size
        preds = predictions.view(-1, N, S, T, C)
        variance_per_sample = preds.var(dim=2)# (B, N, T, 2)
        std_error_per_sample = torch.sqrt(variance_per_sample / S)
        variance_per_agent = variance_per_sample.mean(dim=0)  # (N, T, 2)
        std_error_per_agent = std_error_per_sample.mean(dim=0)  # (N, T, 2) #variance between samples - per each time step and agent
        variance_samples = preds.var(dim=3).mean(dim=0)  # (N, S, 2) varince inside samples - between time steps per agetn. sample diversity
        diversity_score = variance_samples.var(dim=1) #How "spread out" the samples are per agent. N, 2


        for agent in range(N):
            trajectory_wins[agent] += true_indices[:, agent, :].sum(axis=0)

        if iter_num % args.iternum_print == 0:
            print(
                'Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f} | my loss: {:03f}'
                .format(epoch, args.num_epochs, iter_num, total_iter_num, total_loss.item(), loss_pred, loss_recover,
                        loss_kl, loss_diverse, my_loss))
            print("diversity_score", diversity_score)
        iter_num += 1
        all_loss += total_loss.item()

    plot_variance(T, N, variance_per_agent)
    validate_loss = validate(validation_loader)
    scheduler.step()
    model.step_annealer()
    avg_loss = all_loss / iter_num
    plot_agent_trajectory_histogram(3, trajectory_wins, S)

    return avg_loss, validate_loss, diversity_score

def plot_agent_trajectory_histogram(agent_id, trajectory_wins, S):
    if agent_id not in trajectory_wins:
        print(f"Agent {agent_id} not found!")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(range(S), trajectory_wins[agent_id], color='skyblue', edgecolor='black')
    plt.xlabel("Trajectory Index")
    plt.ylabel("Win Count")
    plt.title(f"Trajectory Selection Histogram for Agent {agent_id}")
    plt.xticks(range(S))
    plt.show()



def ploting_losses(train_loss, valid_loss, file_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss[1:], label='Training Loss', color='blue')
    plt.plot(valid_loss[1:], label='Validation Loss', color='red')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()

def ploting_diversity(epochs, N,avg_diversity_np):
    epoch_range = np.arange(epochs)
    width = 0.08

    fig, ax = plt.subplots(figsize=(12, 6))

    for agent in range(N):
        x_pos = epoch_range + (agent - N / 2) * (2 * width)
        ax.bar(x_pos, avg_diversity_np[:, agent, 0], width=width, label=f'Agent {agent} X')
        ax.bar(x_pos + width, avg_diversity_np[:, agent, 1], width=width, label=f'Agent {agent} Y')

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Diversity Score Variance")
    ax.set_title("Diversity Score Variance per Agent Over Epochs")
    ax.legend(ncol=4, fontsize=8, loc='upper right')
    plt.xticks(epoch_range)
    plt.savefig("diversity_plot.png")
    plt.show()
    plt.close()

""" model & optimizer """
model = GroupNet(args, device)
print("params model", model.parameters())
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

""" dataloader """
train_set = FISHDataset(
    obs_len=args.past_length,
    pred_len=args.future_length,
    training=True)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=seq_collate,
    pin_memory=True)

validation_set = FISHDataset(
    obs_len=args.past_length,
    pred_len=args.future_length,
    training=False,
    use_validation=True, validation_split=0.1
)

validation_loader = DataLoader(
    validation_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=seq_collate,
    pin_memory=True
)

""" Loading if needed """
if args.epoch_continue > 0:
    checkpoint_path = os.path.join(args.model_save_dir, str(args.epoch_continue) + '.p')
    print('load model from: {checkpoint_path}')
    model_load = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(model_load['model_dict'])
    if 'optimizer' in model_load:
        optimizer.load_state_dict(model_load['optimizer'])
    if 'scheduler' in model_load:
        scheduler.load_state_dict(model_load['scheduler'])

""" start training """
model.set_device(device)

avg_validate_losses = []
avg_train_losses = []
avg_diversity = []
for epoch in range(args.epoch_continue, args.num_epochs):
    avg_loss, validate_loss, diversity_score = train(train_loader, epoch, args.batch_size, device)
    avg_validate_losses.append(validate_loss)
    avg_train_losses.append(avg_loss)
    avg_diversity.append(diversity_score)

    """ save model """
    if (epoch + 1) % args.model_save_epoch == 0:
        model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'model_cfg': args}
        saved_path = os.path.join(args.model_save_dir, str(epoch + 1) + '.p')
        torch.save(model_saved, saved_path)

ploting_losses(avg_train_losses, avg_validate_losses, 'training_validation_loss.png')
ploting_diversity(args.num_epochs, 8,avg_diversity)
