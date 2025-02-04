import random
import torch
import numpy as np
import matplotlib.lines as mlines
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt


def simulate_separate_controlled_uncontrolled(model, length, steps_control, steps_uncontrol, method, test_loader, args, number_of_agents=None, collective_choose=False):
    # length in seconds!
    # need to send with batch = 1
    total_steps = length / 0.4
    # each step is 0.4 seconds
    random.seed(42)
    num_batches = len(test_loader)

    random_batch_idx = random.randint(0, num_batches - 1)

    sample = test_loader.dataset[0][0].unsqueeze(0)  # B,N, T, 2
    # print("sample", sample)
    centroids_mean = sample[0].cpu().numpy().mean(axis=0)  # T, 2
    centroids_new = sample[0].cpu().numpy().mean(axis=0)
    # print("centroids", centroids.shape)
    target = [80, 90]

    iter = 0
    simulated = np.array(sample[0]) #8 agents, 5 time steps, 2 coords
    while len(simulated[0]) - 5 < total_steps:
        with torch.no_grad():
            prediction = model.inference_simulator(sample)
        # print("prediction", prediction.shape) # (20, N,T,2) -> 20, 8, 10, 2

        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu())  # (20, N,T,2)

        if method == 'mean':
            new_step = np.mean(prediction[:, :, :steps, :], axis=0)
            # print("new_step_old", new_step) # N,T, 2
            mean_centroid = np.mean(new_step, axis=0)  # T, 2
            centroids_mean = np.concatenate((centroids_mean, mean_centroid), axis=0)

        if number_of_agents:
            if collective_choose:
                new_agents = prediction[:, :number_of_agents, :steps, :]  # 20, num, T, 2
                # print("new_agents", new_agents)
                distances = np.linalg.norm(new_agents - target, axis=-1)  # 20, N, T

                distance_scores = distances.sum(axis=(1, 2))  # 20
                closest_indices = np.argsort(distance_scores)[:1]
                closest_positions = new_agents[closest_indices, ...].squeeze(0)

                # print("closest_positions", closest_positions.shape)

            else:
                # if choosing for each time step and each agent seperatly
                new_agents = prediction[:, :number_of_agents, :steps, :]
                distances = np.linalg.norm(new_agents - target, axis=-1)  # 20, N, T
                # Sum distances over all time steps (T)-> Now shape is (20, N)
                total_distances = distances.sum(axis=-1)
                best_indices = total_distances.argmin(axis=0)  # N
                # print("best_indices", best_indices)

                a_idx = np.arange(best_indices.shape[0])
                # t_idx = np.arange(best_indices.shape[1])[None, :]
                # print("a_idx", t_idx.shape,t_idx )
                closest_positions = new_agents[best_indices, a_idx, :, :]  # N, T, 2

            if np.random.rand(1) > 0.5:
                new_step[:number_of_agents, :, :] = closest_positions  # replacing the new agents
            # print("closest_positions",  closest_positions)
            # print("new_step", new_step)

            mean_centroid_new = np.mean(new_step, axis=0)  # T, 2
            centroids_new = np.concatenate((centroids_new, mean_centroid_new), axis=0)

        if method == 'random':
            random_idx = random.randint(0, 19)
            new_step = prediction[random_idx, :, :steps, :]
            mean_centroid = np.mean(new_step, axis=0)  # T, 2
            centroids_mean = np.concatenate((centroids_mean, mean_centroid), axis=0)

        new_trajectory = np.concatenate((sample[0], new_step), axis=1)  # N, T+step, 2
        sample = torch.from_numpy(new_trajectory[:, -5:, :]).unsqueeze(0)  # add batch
        simulated = np.concatenate((simulated, new_step), axis=1)
        # if iter == 10:
        #     print("simulated", simulated)
        iter += 1
    return simulated, centroids_mean, centroids_new


def simulate(model, length, steps, method, test_loader, args, number_of_agents=None, collective_choose=False):
    # length in seconds!
    # need to send with batch = 1
    total_steps = length / 0.4
    # each step is 0.4 seconds
    random.seed(42)
    num_batches = len(test_loader)

    random_batch_idx = random.randint(0, num_batches - 1)

    sample = test_loader.dataset[0][0].unsqueeze(0)  # B,N, T, 2
    # print("sample", sample)
    centroids_mean = sample[0].cpu().numpy().mean(axis=0)  # T, 2
    centroids_new = sample[0].cpu().numpy().mean(axis=0)
    # print("centroids", centroids.shape)
    target = [80, 90]

    iter = 0
    simulated = np.array(sample[0]) #8 agents, 5 time steps, 2 coords
    while len(simulated[0]) - 5 < total_steps:
        with torch.no_grad():
            prediction = model.inference_simulator(sample)
        # print("prediction", prediction.shape) # (20, N,T,2) -> 20, 8, 10, 2

        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu())  # (20, N,T,2)

        if method == 'mean':
            new_step = np.mean(prediction[:, :, :steps, :], axis=0)
            # print("new_step_old", new_step) # N,T, 2
            mean_centroid = np.mean(new_step, axis=0)  # T, 2
            centroids_mean = np.concatenate((centroids_mean, mean_centroid), axis=0)

        if number_of_agents:
            if collective_choose:
                new_agents = prediction[:, :number_of_agents, :steps, :]  # 20, num, T, 2
                # print("new_agents", new_agents)
                distances = np.linalg.norm(new_agents - target, axis=-1)  # 20, N, T

                distance_scores = distances.sum(axis=(1, 2))  # 20
                closest_indices = np.argsort(distance_scores)[:1]
                closest_positions = new_agents[closest_indices, ...].squeeze(0)

                # print("closest_positions", closest_positions.shape)

            else:
                # if choosing for each time step and each agent seperatly
                new_agents = prediction[:, :number_of_agents, :steps, :]
                distances = np.linalg.norm(new_agents - target, axis=-1)  # 20, N, T
                # Sum distances over all time steps (T)-> Now shape is (20, N)
                total_distances = distances.sum(axis=-1)
                best_indices = total_distances.argmin(axis=0)  # N
                # print("best_indices", best_indices)

                a_idx = np.arange(best_indices.shape[0])
                # t_idx = np.arange(best_indices.shape[1])[None, :]
                # print("a_idx", t_idx.shape,t_idx )
                closest_positions = new_agents[best_indices, a_idx, :, :]  # N, T, 2

            if np.random.rand(1) > 0.5:
                new_step[:number_of_agents, :, :] = closest_positions  # replacing the new agents
            # print("closest_positions",  closest_positions)
            # print("new_step", new_step)

            mean_centroid_new = np.mean(new_step, axis=0)  # T, 2
            centroids_new = np.concatenate((centroids_new, mean_centroid_new), axis=0)

        if method == 'random':
            random_idx = random.randint(0, 19)
            new_step = prediction[random_idx, :, :steps, :]
            mean_centroid = np.mean(new_step, axis=0)  # T, 2
            centroids_mean = np.concatenate((centroids_mean, mean_centroid), axis=0)

        new_trajectory = np.concatenate((sample[0], new_step), axis=1)  # N, T+step, 2
        sample = torch.from_numpy(new_trajectory[:, -5:, :]).unsqueeze(0)  # add batch
        simulated = np.concatenate((simulated, new_step), axis=1)
        # if iter == 10:
        #     print("simulated", simulated)
        iter += 1
    return simulated, centroids_mean, centroids_new


def visualize_simulation(model, length, steps, method, test_loader, args, fps, output_path, test_simulation=False,
                         agent_number=None):
    if test_simulation:
        # print("dataset exp", test_loader.dataset.__len__())
        simulated = test_loader.dataset[0][0]  # N, T, 2
        print("simulated out", simulated)
        for i in range(int(length // 2)):
            five_time_steps = test_loader.dataset[i + 1][0]
            if i < 2:
                print("five_time_steps", five_time_steps)
            simulated = np.concatenate((simulated, five_time_steps), axis=1)
            centroids = simulated.mean(axis=0)  # T, 2
    else:
        simulated, centroids, centroids_new = simulate(length, steps, method, test_loader, args, agent_number)
        # print(simulated.shape, centroids.shape, centroids_new.shape)
    N, T, _ = simulated.shape

    fig, ax = plt.subplots(figsize=(8, 6))
    x_min, x_max = simulated[:, :, 0].min() - 1, simulated[:, :, 0].max() + 1
    y_min, y_max = simulated[:, :, 1].min() - 1, simulated[:, :, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Simulation of agents")

    colors = plt.cm.tab10(np.linspace(0, 1, N + 1))
    scatter = ax.scatter([], [], s=80)
    scatter2 = ax.scatter([], [], s=80, marker='x')
    if agent_number is not None:
        scatter3 = ax.scatter([], [], s=80, marker='+', alpha=0.7)
        # colors = plt.cm.tab10(np.linspace(0, 1, N + 2))

    def init():
        scatter.set_offsets([])
        return (scatter,)

    def update(frame):
        # in current frame get the positions for all agents.
        current_positions = simulated[:, frame, :]  # N, 2
        current_centre = centroids[frame, :]  # 2
        if agent_number is not None:
            new_centre = centroids_new[frame, :]
            scatter3.set_offsets(new_centre)
            scatter3.set_color(colors[-2])

        # update plot
        scatter.set_offsets(current_positions)
        scatter.set_color(colors)
        scatter2.set_offsets(current_centre)
        scatter2.set_color(colors[-1])

        # title showing the current time step.
        ax.set_title(f"Simulation at time : {frame * 0.4:.1f}/{T * 0.4:.1f}")
        return (scatter,)

    ani = animation.FuncAnimation(
        fig, update, frames=range(T),
        init_func=init, blit=True, interval=fps
    )

    # save
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)