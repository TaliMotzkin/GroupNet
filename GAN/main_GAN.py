import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from config import parse_args
from torch import nn
from models import Generator, Mission, Discrimiter
import sys
from utilis_GAN import saveModel
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data.dataloader_fish import FISHDataset, seq_collate
from torch.utils.data import DataLoader
import numpy as np
import random
from model.GroupNet_nba import GroupNet
from loss import LossCompute
from dataloader_GAN import TrajectoryDataset

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def create_traj(test_loader, model, length, args, real_data =False):

    sample = test_loader.dataset[:args.batch_size][0].to(args.device) # B,N, T, 2
    iter = 0
    past_traj = torch.empty((0, 8, 5,2)).to(args.device) # total samples, 8 agents, 5 time steps, 2 coords

    future_traj = torch.empty((0, 8, 10, 2, 20)).to(args.device)  # (Total_samples, N, 10, 2, 20)
    selected_traj = torch.empty((0, 8, 10, 2)).to(args.device)  # (Total_samples, N, 10, 2)
    H_list = torch.empty((0, 9,8)).to(args.device)

    while len(past_traj) - 5 < length:
        with torch.no_grad():
            prediction, H = model.inference_simulator(sample) # (20, B*N,10,2)

        if args.method == 'mean':
            agents_future_steps = torch.mean(prediction, dim=0)# ( B*N,10,2)
        elif args.method == 'first':
            agents_future_steps = prediction[0, :, :, :]# ( B*N,10,2)
        agents_future_steps = agents_future_steps.view(args.batch_size, 8, 10, 2)
        past_traj = torch.cat((past_traj, sample), dim=0)  # Add new timestep data
        sample = agents_future_steps[:, :, 5:, :]


        prediction = prediction.permute(1, 2, 3, 0).view(args.batch_size, 8, 10, 2, 20)


        future_traj = torch.cat((future_traj, prediction), dim=0)

        selected_traj = torch.cat((selected_traj, agents_future_steps), dim=0)
        H_list = torch.cat((H_list, H), dim=0)
        iter += 1

    traj_dataset = TrajectoryDataset(past_traj, future_traj, selected_traj, H_list)

    return traj_dataset



def plot_losses(args, train_losses_g, train_losses_d, val_losses_g, val_losses_d,
                train_scores_real, train_scores_fake, val_scores_real, val_scores_fake):
    epochs = range(1, len(train_losses_g) + 1)

    train_scores_real = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in train_scores_real])
    train_scores_fake = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in train_scores_fake])
    val_scores_real = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in val_scores_real])
    val_scores_fake = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in val_scores_fake])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses_g, label='Train Generator Loss', color='blue')
    plt.plot(epochs, val_losses_g, label='Val Generator Loss', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Generator Loss')
    plt.legend()
    plt.title('Generator Loss Progression')
    plt.savefig(f"GAN\GAN_plots\GAN_Generator_Loss_{args.timestamp}.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses_d, label='Train Discriminator Loss', color='red')
    plt.plot(epochs, val_losses_d, label='Val Discriminator Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator Loss')
    plt.legend()
    plt.title('Discriminator Loss Progression')
    plt.savefig(f"GAN\GAN_plots\GAN_Discriminator_Loss_{args.timestamp}.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_scores_real, label='Train Real Score', color='green')
    plt.plot(epochs, train_scores_fake, label='Train Fake Score', color='purple')
    plt.plot(epochs, val_scores_real, label='Val Real Score', color='lime')
    plt.plot(epochs, val_scores_fake, label='Val Fake Score', color='magenta')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator Score')
    plt.legend()
    plt.title('Discriminator Scores Over Training')
    plt.savefig(f"GAN\GAN_plots\GAN_Discriminator_Scores_{args.timestamp}.png")
    plt.show()

def train(train_loader,val_loader, args, G, M, D):
    lossfn = LossCompute(G, D, M,  args)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    optimizer_M = torch.optim.Adam(M.parameters(), lr=args.lr)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_M = torch.optim.lr_scheduler.StepLR(optimizer_M, step_size=args.lr_step, gamma=args.lr_gamma)

    # Track losses
    train_losses_g = []
    train_losses_d = []
    val_losses_g = []
    val_losses_d = []
    train_scores_real = []
    train_scores_fake = []
    val_scores_real = []
    val_scores_fake = []

    iter_num = 0
    for i in range(args.epoch):
        G.train()
        D.train()
        M.train()
        train_loss_g = 0
        train_loss_d = 0
        train_real_score = 0
        train_fake_score = 0
        num_batches = 0
        for batch in train_loader:
            past = batch['past_traj']
            prediction = batch['group_net']
            selected = batch['selected_traj']
            H = batch['H_list']

            iter_num += 1
            num_batches += 1

            # past = past.view(args.batch_size*8, 5, 2)
            # print("agents_future_steps", agents_future_steps.shape)
            # agents_future_steps = torch.tensor(agents_future_steps, dtype=torch.float32, device=args.device)
            final_positions = selected[:,args.agent, -1, :]  # Shape (B, 2)

            target_tensor = torch.tensor(args.target, dtype=torch.float32, device= args.device)

            distances = torch.norm(final_positions - target_tensor, dim=-1)  # Shape (B,)
            mission = (distances <= 2).float()

            optimizer_M.zero_grad()


            prediction = prediction.view(20, args.batch_size *8, 10, 2)  # (20, N,T,2)
            loss_m = lossfn.compute_Mission_loss(past, selected,args.target,mission, H)
            loss_m.backward()
            optimizer_M.step()

            optimizer_D.zero_grad()
            loss_d, loss_real, loss_fake, scores_fake, scores_real = lossfn.compute_discriminator_loss(prediction, H , past, mission , args.agent,  args.target, selected)
            loss_d.backward()
            optimizer_D.step()


            optimizer_G.zero_grad()
            loss_g_all, loss_g_l2, loss_g, col_loss = lossfn.compute_generator_loss(prediction, H , past, mission , args.agent,  args.target, selected)
            loss_g_all.backward()
            optimizer_G.step()

            train_loss_d += loss_d.item()
            train_loss_g += loss_g_all.item()
            scores_real_mean = scores_real.mean(dim=(1, 2)).detach().cpu().numpy()  # (B*N, 15, 1) -> (B,)
            scores_fake_mean = scores_fake.mean(dim=(1, 2)).detach().cpu().numpy()  # (B*N, 15, 1) -> (B,)

            train_real_score += scores_real_mean.mean()
            train_fake_score += scores_fake_mean.mean()
            break

        scheduler_G.step()
        scheduler_D.step()
        scheduler_M.step()
                # print(output) #64, 10, 2

        # Store training metrics
        train_losses_g.append(train_loss_g / num_batches)
        train_losses_d.append(train_loss_d / num_batches)
        train_scores_real.append(train_real_score / num_batches)
        train_scores_fake.append(train_fake_score / num_batches)

        # Validation Phase
        G.eval()
        D.eval()
        M.eval()
        val_loss_g = 0
        val_loss_d = 0
        val_real_score = 0
        val_fake_score = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                past = batch['past_traj']
                prediction = batch['group_net']
                selected = batch['selected_traj']
                H = batch['H_list']

                num_batches += 1

                final_positions = selected[:, args.agent, -1, :]
                target_tensor = torch.tensor(args.target, dtype=torch.float32, device=args.device)
                distances = torch.norm(final_positions - target_tensor, dim=-1)
                mission = (distances <= 2).float()

                prediction = prediction.view(20, args.batch_size * 8, 10, 2)
                loss_d, loss_real, loss_fake , scores_fake, scores_real= lossfn.compute_discriminator_loss(prediction, H, past, mission,
                                                                                 args.agent, args.target, selected)
                loss_g_all, loss_g_l2, loss_g, col_loss = lossfn.compute_generator_loss(prediction, H, past,
                                                                                        mission, args.agent,
                                                                                        args.target, selected)

                val_loss_d += loss_d.item()
                val_loss_g += loss_g_all.item()
                scores_real_mean = scores_real.mean(dim=(1, 2)).detach().cpu().numpy()  # (B*N, 15, 1) -> (B,)
                scores_fake_mean = scores_fake.mean(dim=(1, 2)).detach().cpu().numpy()  # (B*N, 15, 1) -> (B,)

                val_real_score += scores_real_mean.mean()
                val_fake_score += scores_fake_mean.mean()

        # Store validation metrics
        val_losses_g.append(val_loss_g / num_batches)
        val_losses_d.append(val_loss_d / num_batches)
        val_scores_real.append(val_real_score / num_batches)
        val_scores_fake.append(val_fake_score / num_batches)

        # Save model checkpoint every epoch
        if (i + 1) % args.save_every == 0:
            saveModel(G, D, M, args, str(i + 1))

        print(
            f"Epoch [{i + 1}/{args.epoch}] - Train Loss G: {train_losses_g[-1]:.4f}, D: {train_losses_d[-1]:.4f} | Val Loss G: "
            f"{val_losses_g[-1]:.4f}, D: {val_losses_d[-1]:.4f}")

    plot_losses(args, train_losses_g, train_losses_d, val_losses_g, val_losses_d, train_scores_real, train_scores_fake,val_scores_real, val_scores_fake)

def plot_score_list(score_list, args):
    plt.figure(figsize=(8, 5))
    plt.plot(score_list, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Score vs Iteration")
    plt.grid(True)
    plt.savefig(f"GAN\score_list_{args.method}.png")

def vis_predictions(future_traj):
    N, T, _ = future_traj.shape

    # Set up the figure and axis limits
    fig, ax = plt.subplots(figsize=(8, 6))
    x_min = future_traj[:, :, 0].min() - 1
    x_max = future_traj[:, :, 0].max() + 1
    y_min = future_traj[:, :, 1].min() - 1
    y_max = future_traj[:, :, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Agent Trajectories Simulation")

    # Determine the color for each agent: args.agent (or agents) will be red, others pink.
    if isinstance(args.agent, (list, tuple)):
        agent_indexes = args.agent
    else:
        agent_indexes = [args.agent]
    colors = ['red' if i in agent_indexes else 'pink' for i in range(N)]

    # Plot the target coordinates as an 'X'
    target_x, target_y = args.target[0], args.target[1]
    ax.scatter([target_x], [target_y], color='black', marker='x', s=100, label='Target')

    # Create an empty scatter plot to update with agent positions.
    scatter = ax.scatter([], [], s=80)

    def init():
        scatter.set_offsets([])
        return scatter,

    def update(frame):
        # For the current frame, get positions of all agents
        current_positions = future_traj[:, frame, :]  # shape: (N, 2)
        scatter.set_offsets(current_positions)
        scatter.set_color(colors)
        ax.set_title(f"Simulation at time: {frame * 0.4:.1f}s / {T * 0.4:.1f}s")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=range(T), init_func=init, blit=True, interval=10)
    writer = PillowWriter(fps=10)
    ani.save(f"GAN/simulation_{args.method}_{args.timestamp}.gif", writer=writer)
    plt.close(fig)

def pred(test_loader, args, model, G, D):

    G.eval()
    D.eval()
    model.eval()
    iter_num = 0
    sample = test_loader.dataset[2000][0].unsqueeze(0).to(args.device) # B, N, T, 2
    total_steps = args.length / 0.4
    future_traj = np.array(sample[0].detach().cpu().numpy())   #  8 agents, 5 time steps, 2 coords
    score_list = []

    while len(future_traj[0]) - 5 < total_steps:
        with torch.no_grad():
            prediction, H = model.inference_simulator(sample) #20, BN, T, 2

        if args.method == 'mean':
            agents_future_steps = torch.mean(prediction[:, :, :10, :], dim=0)
        elif args.method == 'first':
            agents_future_steps = prediction[0, :, :10, :]

        # print("agents_future_steps", agents_future_steps.shape)

        final_positions = agents_future_steps.view(1, 8, 10, 2)[:,args.agent, -1, :]  # Shape (B, 2)
        mission = torch.ones(final_positions.shape[0], device=args.device) # Encourage generator to produce data that is getting closer to the mission

        past_data = sample
        pred_trajectories = G(prediction, H, past_data, mission,args.agent, args.target)

        agents_future_steps_GEN = agents_future_steps.view(1, 8, 10, 2)
        agents_future_steps_GEN[:, args.agent, :, :] = pred_trajectories


        scores = D(prediction, H, past_data, args.agent, agents_future_steps_GEN)

        # Adding the new trajectory and scores from the discriminator
        future_traj = np.concatenate((future_traj, agents_future_steps_GEN.squeeze(0).detach().cpu().numpy()), axis=1) #N, T, 2
        score_list.append(scores.mean(dim=(1, 2)).detach().cpu().numpy())
        sample = agents_future_steps_GEN[:, :, 5:, :] #taking the 5 last time steps for next prediction
        iter_num += 1
    plot_score_list(score_list, args)
    vis_predictions(future_traj)


def load_dataset(test_loader, args, model):
    DATASET_PATH = f"trajectory_dataset_{args.method}_{args.length}.pt"

    if os.path.exists(DATASET_PATH):
        print("Loading existing dataset...")
        traj_dataset = torch.load(DATASET_PATH)  # Load dataset
    else:
        print("Creating new dataset...")
        traj_dataset = create_traj(test_loader, model, args.length, args, real_data=False)
        torch.save(traj_dataset, DATASET_PATH)

    # Split into train and validation sets
    total_size = len(traj_dataset)
    # print("total_size", total_size)

    val_size = int(0.2 * total_size)  # 20% for validation
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(traj_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader

if __name__ == '__main__':
    args = parse_args()

    """ setup """
    names = [x for x in args.model_names.split(',')]



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

        model = GroupNet(training_args, args.device)
        model.set_device(args.device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)

        G = Generator(args.device, args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len,
                      args.dropout, 9).to(args.device)

        M = Mission(args.device, args.dim, args.mlp_dim, args.depth, args.heads, args.dropout, 9).to(args.device)
        D = Discrimiter(args.device, args.dim, args.mlp_dim, args.depth, args.heads, args.dropout, 9).to(args.device)

        if args.mode == 'train':

            # Creating new data sets based on the first initialization of the test data, then it is just generated with groupnet, therefore, there is no mixing of test-train,
            #but for in case - I will start testing on differnt timestep
            train_dataset, val_dataset = load_dataset(test_loader, args, model)
            train(train_dataset, val_dataset, args, G, M, D)

        else:


            G_path = f"{args.GAN_models}/G_raw_1.pth"
            D_path = f"{args.GAN_models}/D_raw_1.pth"
            G.load_state_dict(torch.load(G_path))
            D.load_state_dict(torch.load(D_path))

            pred(test_loader, args, model, G, D)


