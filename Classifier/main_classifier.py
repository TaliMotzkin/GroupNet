import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from data.dataloader_fish import FISHDataset, seq_collate
from torch.utils.data import DataLoader
import numpy as np
import random
from model.GroupNet_nba import GroupNet
from Simulator import *
import torch.optim as optim
import torch
import torch.nn as nn
from config_classifier import parse_args

class TrajectoryClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        """
        Classifies whether a given trajectory sequence belongs to a controlled (real) or random (fake) movement.

        input_dim: 2 (X, Y coordinates)
        hidden_dim: LSTM hidden size
        num_layers: Number of LSTM layers
        """
        super(TrajectoryClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (Batch, N, Seq, 2) - Trajectories for all agents
        """
        batch_size, num_agents, seq_length, _ = x.shape
        x = x.view(batch_size * num_agents, seq_length, -1)
        _, (hidden, _) = self.lstm(x)  #
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  #
        hidden = hidden.view(batch_size, num_agents, -1)
        x = hidden.mean(dim=1)
        x = self.fc(x)
        return x



def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            trajectories, labels = batch
            trajectories, labels = trajectories.to(device), labels.to(device)

            outputs = model(trajectories).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches  # Return average validation loss

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            trajectories, labels = batch
            trajectories, labels = trajectories.to(device), labels.to(device)

            outputs = model(trajectories).squeeze()
            predictions = (outputs > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.shape[0]

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def plot_losses(args, train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"Classifier/loss_calssifier_{args.method}_{args.length}_many_targets_fish_splited_train_test.png")

def train(args, model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            trajectories, labels = batch
            # print("trajectories", trajectories.shape[0])
            trajectories, labels = trajectories.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(trajectories).squeeze()

            # print("outputs", outputs)
            # print("lanel, ", labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = validate(model, criterion, val_loader, device)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Plot Losses
    plot_losses(args, train_losses, val_losses)



class TrajectoryDataset(Dataset):
    def __init__(self, real_trajectories, fake_trajectories, seq_length = 20):
        """
        real_trajectories: Simulated data with controlled agents
        fake_trajectories: Simulated data with all random agents
        """
        self.seq_length = seq_length
        self.trajectories = []
        self.labels = []

        self._split_and_store(real_trajectories, label=1)

        # Process fake trajectories (label 0)
        self._split_and_store(fake_trajectories, label=0)

    def _split_and_store(self, trajectory, label):
        """
        Splits a single trajectory into chunks of `seq_length` time steps.
        """
        N, T, _ = trajectory.shape  # (N agents, T time steps, 2D coordinates)
        for start in range(0, T - self.seq_length + 1, self.seq_length):  # Non-overlapping windows
            segment = trajectory[:, start:start + self.seq_length, :]  # Shape (N, seq_length, 2)
            self.trajectories.append(segment)
            self.labels.append(label)


    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return torch.tensor(self.trajectories[idx], dtype=torch.float32), torch.tensor(self.labels[idx],
                                                                                           dtype=torch.float32)
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Splits should sum to 1!"

    total_size = len(dataset)
    # print("total_size", total_size)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    # test_size = total_size - train_size - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])
    # print(train_set[0], val_set[0], test_set[0])
    return train_set, val_set
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

        dataset_train_path = f"Classifier/trajectories_{args.method}_{args.length}_many_targets_fish_train.pt"
        dataset_test_path = f"Classifier/trajectories_{args.method}_{args.length}_many_targets_fish_test.pt"

        if os.path.exists(dataset_train_path):
            print(f"Loading existing dataset from {dataset_train_path}...")
            data = torch.load(dataset_train_path)
            real, fake_list = data["real"], data["fake"]
        else:
            init_sample = 0
            print(f"No dataset found. Creating new dataset...")
            real, _, _ = simulate(init_sample, args.target, model, args.length, 1, args.method, test_loader, args, number_of_agents=None, collective_choose=False)

            fake_list = []
            for target in args.target:

                fake , _, _ =simulate_separate_controlled_uncontrolled(init_sample, target, args.heat_map_path, model, args.length/5, 1,
                                                          1, args.method, test_loader, args, [0, 1, 2],
                                                          "closest_centroid")
                fake_list.append(fake)
            fake_list = np.concatenate(fake_list, axis=1)
            torch.save({"real": real, "fake": fake_list}, dataset_train_path)
            print(f"Dataset saved at {dataset_train_path}")

        if os.path.exists(dataset_test_path):
            print(f"Loading existing dataset from {dataset_test_path}...")
            data_test = torch.load(dataset_test_path)
            real_test, fake_list_test = data_test["real"], data_test["fake"]
        else:
            init_sample = 1200
            print(f"No dataset found. Creating new dataset...")
            real_test, _, _ = simulate(init_sample, args.test_target, model, args.length/4, 1, args.method, test_loader, args, number_of_agents=None, collective_choose=False)

            fake_list_test = []
            for target in args.test_target:

                fake_test , _, _ =simulate_separate_controlled_uncontrolled(init_sample, target, args.heat_map_path, model, args.length/20, 1,
                                                          1, args.method, test_loader, args, [0, 1, 2],
                                                          "closest_centroid")
                fake_list_test.append(fake_test)
            fake_list_test = np.concatenate(fake_list_test, axis=1)
            torch.save({"real": real_test, "fake": fake_list_test}, dataset_test_path)
            print(f"Dataset saved at {dataset_test_path}")

        """ Create Dataset and DataLoaders """
        dataset_train = TrajectoryDataset(real, fake_list)
        dataset_test = TrajectoryDataset(real_test, fake_list_test)

        train_set, val_set = split_dataset(dataset_train)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

        """ Initialize Classifier """
        model_classifier = TrajectoryClassifier().to(args.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model_classifier.parameters(), lr=0.001)

        """ Train and Evaluate """
        train(args ,model_classifier, optimizer, criterion, train_loader, val_loader, args.epoch, args.device)
        test(model_classifier, test_loader, args.device)
