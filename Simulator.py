import random
import torch
import numpy as np
import matplotlib.lines as mlines
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import os

def generate_heatmaps(output_path, simulated, best_simulated, worst_simulated, target):
    """
    Generate heatmaps of visitation frequency in XY space for best and worst options.

    :param simulated: The final simulated trajectory
    :param best_simulated: The best-selected trajectory
    :param worst_simulated: The worst-selected trajectory
    """

    def adjust_bins(data, target, centre, num_bins=50):
        if centre:
            x_min, x_max = data[ :, 0].min(), data[ :, 0].max()
            y_min, y_max = data[ :, 1].min(), data[:, 1].max()
        else:
            x_min, x_max = data[:, :, 0].min(), data[:, :, 0].max()
            y_min, y_max = data[:, :, 1].min(), data[:, :, 1].max()

        x_edges = np.linspace(x_min, x_max, num_bins + 1)
        y_edges = np.linspace(y_min, y_max, num_bins + 1)

        # Ensure target is well within a bin, not on the edge
        x_edges = np.sort(np.unique(np.append(x_edges, target[0] - np.diff(x_edges[:2])[0] / 2)))
        y_edges = np.sort(np.unique(np.append(y_edges, target[1] - np.diff(y_edges[:2])[0] / 2)))

        return x_edges, y_edges

    def plot_heatmap(data, title, target):
        x_coords = data[:, :, 0].flatten()
        y_coords = data[:, :, 1].flatten()

        x_edges, y_edges = adjust_bins(data, target, False)

        heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])

        # Find target frequency
        x_idx = np.digitize(target[0], x_edges) - 1
        y_idx = np.digitize(target[1], y_edges) - 1
        print("XY", x_idx, y_idx)
        print("edges", x_edges, y_edges)
        target_frequency = heatmap[x_idx, y_idx] if (
                    0 <= x_idx < heatmap.shape[0] and 0 <= y_idx < heatmap.shape[1]) else 0


        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        plt.colorbar(label="Frequency")

        # Plot the target location with an 'X'
        plt.scatter(target[0], target[1], color='cyan', marker='x', s=100, label=f"Target ({target[0]}, {target[1]})")

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        # Annotate the target frequency near the legend or title
        plt.title(f"{title}\n(Target Frequency: {int(target_frequency)})", fontsize=14)
        path = os.path.splitext(output_path)[0]
        filename = title.replace(" ", "_") + path +".png"
        plt.savefig(filename,  dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_heatmap_centroids(data, title, target):
        x_coords = data[:, 0]
        y_coords = data[:, 1]

        x_edges, y_edges = adjust_bins(data, target, True)

        heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])

        # Find target frequency
        x_idx = np.digitize(target[0], x_edges) - 1
        y_idx = np.digitize(target[1], y_edges) - 1
        print("target", target[0])
        print("XY", x_idx, y_idx)
        print("edges", x_edges, y_edges)
        target_frequency = heatmap[x_idx, y_idx] if (
                    0 <= x_idx < heatmap.shape[0] and 0 <= y_idx < heatmap.shape[1]) else 0

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        plt.colorbar(label="Frequency")

        # Plot the target location with an 'X'
        plt.scatter(target[0], target[1], color='cyan', marker='x', s=100, label=f"Target ({target[0]}, {target[1]})")

        # Annotate the target frequency near the legend or title
        plt.title(f"{title}\n(Target Frequency: {int(target_frequency)})", fontsize=14)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Save the figure
        path = os.path.splitext(output_path)[0]
        filename = title.replace(" ", "_") + path + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()



    # Compute centroids for heatmap visualization
    best_centroids = np.mean(best_simulated, axis=0)  # (T, 2)
    worst_centroids = np.mean(worst_simulated, axis=0)  # (T, 2)

    plot_heatmap_centroids(best_centroids, "Best Option Centroids Heatmap of Visitation", target)
    plot_heatmap_centroids(worst_centroids, "Worst Option Centroids Heatmap of Visitation", target)
    # Plot heatmaps
    plot_heatmap(best_simulated, "Best Option Heatmap of Visitation", target)
    plot_heatmap(worst_simulated, "Worst Option Heatmap of Visitation", target)


def choose_option(option_results, method):
    """
    Chooses the best and worst option based on the given method.
    Updates the simulated array and prepares the sample for the next iteration.

    :param option_results: Dictionary containing centroid distances for each option.
    :param method: Choosing criteria ('mean_of_centroids', 'closest_centroid', 'final_centroid')
    :return: best_simulated, worst_simulated (selected best and worst options)
    """
    best_option = min(option_results, key=lambda x: option_results[x][method])
    worst_option = max(option_results, key=lambda x: option_results[x][method])

    best_simulated = option_results[best_option]['simulated']
    worst_simulated = option_results[worst_option]['simulated']

    return best_simulated, worst_simulated


def test_option(method, agent_indexes, iter, options_controled, new_step_uncontroled, sample, predicting_times, residual_steps, agent_index, model, args, steps_uncontrol, target, simulated):


    closest_centroid = float("inf")

    for prediction_step in range(predicting_times):

        new_trajectory = np.concatenate((sample[0], new_step_uncontroled), axis=1)  # N, T+step, 2
        sample = torch.from_numpy(new_trajectory[:, -5:, :]).unsqueeze(0)  # add batch
        with torch.no_grad():
            prediction = model.inference_simulator(sample)
        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu())  # (20, N,T,2)

        if method == "mean":
            new_step_uncontroled = np.mean(prediction[:, :, :steps_uncontrol, :], axis=0)  # N,T, 2

        if method == "random":
            random_idx = random.randint(0, 19)
            new_step_uncontroled = prediction[random_idx, :, :steps_uncontrol, :]
        # if iter < 2:
        #     print("iteration in option 0", prediction_step+1, "steps:", options_controled[ :, :])
        #
        #     print("before agent: ", new_step_uncontroled[agent_index, :, :])
        new_step_uncontroled[agent_index, :, :] = options_controled[agent_indexes, :steps_uncontrol, :]  # setting to the chosen option #,N, T, 2
        options_controled = options_controled[:, steps_uncontrol:, :]

        # if iter < 2:
            # print("after agent: ", new_step_uncontroled[agent_index, :, :])


        simulated = np.concatenate((simulated, new_step_uncontroled), axis=1) #N, T, 2

        # centroid calculations
        mean_centroid = np.mean(new_step_uncontroled, axis=0)  # T, 2

        closest_centroid_test = np.min(np.linalg.norm(mean_centroid - target, axis=1))
        if closest_centroid_test < closest_centroid:
            closest_centroid = closest_centroid_test

    if residual_steps > 0:


        new_trajectory = np.concatenate((sample[0], new_step_uncontroled), axis=1)  # N, T+step, 2
        sample = torch.from_numpy(new_trajectory[:, -5:, :]).unsqueeze(0)
        with torch.no_grad():
            prediction = model.inference_simulator(sample)
        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu())
        if method == "mean":
            new_step_uncontroled = np.mean(prediction[:, :, :steps_uncontrol, :], axis=0)  # N,T, 2

        if method == "random":
            random_idx = random.randint(0, 19)
            new_step_uncontroled = prediction[random_idx, :, :steps_uncontrol, :]

        new_step_uncontroled[agent_index, :, :] = options_controled[agent_indexes]  # setting to the chosen option #,N, T, 2 - T shoud mutch ti the residual

        simulated = np.concatenate((simulated, new_step_uncontroled), axis=1)

        mean_centroid = np.mean(new_step_uncontroled, axis=0)  # T, 2

        closest_centroid_test = np.min(np.linalg.norm(mean_centroid - target, axis=1))
        if closest_centroid_test < closest_centroid:
            closest_centroid = closest_centroid_test


    return closest_centroid, simulated



def simulate_separate_controlled_uncontrolled(target_xy, output_path, model, length, steps_control, steps_uncontrol, method, test_loader, args, agent_indexes, choosing_option_by):
    # length in seconds!
    # need to send with batch = 1
    total_steps = length / 0.4
    # each step is 0.4 seconds
    random.seed(42)
    num_batches = len(test_loader)


    sample = test_loader.dataset[1500][0].unsqueeze(0)  # B,N, T, 2
    # print("sample", sample)

    target = target_xy

    simulated = np.array(sample[0]) #8 agents, 5 time steps, 2 coords

    option_results = {}
    iter = 0
    while len(simulated[0]) - 5 < total_steps:


        if steps_uncontrol <= steps_control:
            predicting_times = steps_control//steps_uncontrol

            with torch.no_grad():
                prediction = model.inference_simulator(sample) #-> 20, 8, 10, 2
            prediction = prediction * args.traj_scale
            prediction = np.array(prediction.cpu())  # (20, N,T,2)


            options_controled = prediction[:, :, :steps_control, :] # (20, N,T,2) #stiff predictor
            # print("options_controled shape", options_controled.shape)
            options_controled = np.clip(options_controled, 0, 100)
            # print("options_controled clipped shape", options_controled.shape)

            if method == "mean":
                new_step_uncontroled = np.mean(prediction[:, :, :steps_uncontrol, :], axis=0) # N,T, 2

            if method == "random":
                random_idx = random.randint(0, 19)
                new_step_uncontroled = prediction[random_idx, :, :steps_uncontrol, :]

            residual_steps = steps_control % steps_uncontrol

            for option in range(prediction.shape[0]):

                copy_new_step_uncontroled = new_step_uncontroled.copy()

                # if iter < 2:
                #     # print("num of options", prediction.shape[0])
                #     print("option", option, "steps:", options_controled[option, :, :] )
                #
                #     print("before agent: ", copy_new_step_uncontroled[agent_indexes, :, :])
                copy_new_step_uncontroled[agent_indexes, :, :] = options_controled[option,agent_indexes,:steps_uncontrol,  :]  # setting to the chosen option #,N, T, 2

                # if iter < 2:
                #     print("after agent: ",  copy_new_step_uncontroled[agent_indexes, :, :])
                # centroid calculations
                mean_centroid = np.mean(copy_new_step_uncontroled, axis=0)  # T, 2

                closest_centroid = np.min(np.linalg.norm(mean_centroid - target, axis=1))

                simulated_option = np.concatenate((simulated, copy_new_step_uncontroled), axis=1)

                if predicting_times - 1 > 0:
                    closest_centroid_test, simulated_option = test_option (method, agent_indexes, iter, options_controled[option, :, steps_uncontrol:, :], copy_new_step_uncontroled,
                                                                            sample, predicting_times - 1, residual_steps, agent_indexes, model, args, steps_uncontrol, target, simulated_option)

                    if closest_centroid_test < closest_centroid:
                        closest_centroid = closest_centroid_test
                    # mean_of_centroids = (mean_of_centroids_test + mean_of_centroids) /2

                last_centroid = np.mean(simulated_option[:, -1, :], axis=0)
                last_distance = np.linalg.norm(last_centroid - target)
                centroids_per_t = np.mean(simulated_option, axis=0)  #  (T, 2) -> Mean across N for each T
                mean_of_centroids_distance = np.mean(np.linalg.norm(centroids_per_t - target, axis=1))

                # if iter < 2:
                #     print("simulated option: ", option, simulated_option)
                option_results[option] = {
                    'final_centroid': last_distance,
                    'closest_centroid': closest_centroid,
                    'mean_of_centroids': mean_of_centroids_distance,
                    'simulated': simulated_option
                }

            best_simulated, worst_simulated = choose_option(option_results, choosing_option_by)
            simulated = best_simulated  # Update with best option
            # if iter >145 and iter <149:
            #     print("iter", iter, simulated[0])
            sample = torch.from_numpy(simulated[:, -5:, :]).unsqueeze(0)
            iter += 1

    # print(simulated[1])
    generate_heatmaps(output_path, simulated, best_simulated, worst_simulated, target)
    centroids_mean = np.mean(simulated, axis=0)
    return simulated, centroids_mean, centroids_mean


def simulate(target_xy, model, length, steps, method, test_loader, args, number_of_agents=None, collective_choose=False):
    # length in seconds!
    # need to send with batch = 1
    total_steps = length / 0.4
    # each step is 0.4 seconds
    random.seed(42)
    # print("test_loader", test_loader)
    num_batches = len(test_loader)

    random_batch_idx = random.randint(0, num_batches - 1)

    sample = test_loader.dataset[0][0].unsqueeze(0)  # B,N, T, 2
    # print("sample", sample.shape)
    centroids_mean = sample[0].cpu().numpy().mean(axis=0)  # T, 2
    centroids_new = sample[0].cpu().numpy().mean(axis=0)
    # print("centroids", centroids.shape)
    target = target_xy

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

        if method == 'first':
            new_step = prediction[0, :, :steps, :]
            mean_centroid = np.mean(new_step, axis=0)  # T, 2
            centroids_mean = np.concatenate((centroids_mean, mean_centroid), axis=0)

        new_trajectory = np.concatenate((sample[0], new_step), axis=1)  # N, T+step, 2
        sample = torch.from_numpy(new_trajectory[:, -5:, :]).unsqueeze(0)  # add batch
        simulated = np.concatenate((simulated, new_step), axis=1)
        # if iter == 10:
        #     print("simulated", simulated)
        iter += 1
    return simulated, centroids_mean, centroids_new

def simulate_raw_data_with_agents(model, args, targets, test_loader, agent_indexes, length, step, method):
    # print("dataset exp", test_loader.dataset.__len__())

    sample = test_loader.dataset[0][0].unsqueeze(0)  # B,N, T, 2
    total_steps = length / 0.4
    simulated  = np.array(sample[0])  # 8 agents, 5 time steps, 2 coords
    true_data = np.array(sample[0])

    for i in range(int(length // 2)):
        five_time_steps = test_loader.dataset[i + 1][0]
        true_data = np.concatenate((true_data, five_time_steps), axis=1)
    goal  = [False] *len(agent_indexes)
    goal_status = ["","","","",""]
    iter = 0
    while len(simulated[0]) - 5 < total_steps:
        with torch.no_grad():
            prediction = model.inference_simulator(sample)
        # print("prediction", prediction.shape) # (20, N,T,2) -> 20, 8, 10, 2
        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu())

        new_step = true_data[:, len(simulated[0]):len(simulated[0]) + step, :]
        # if iter <2:
        #     print("sample" , sample[0][0])
        #     print("previous data:" , new_step[0])
        iter_message = ""
        for i, agent in enumerate(agent_indexes):
            if method == "mean":
                seperate_step = np.mean(prediction[:, agent, :step, :], axis=0)
                new_step[agent, :, :] = seperate_step

            # if iter <2:
            #     print("agents indx", agent)
            #     print(new_step.flags.writeable)  # Should print True
            #     print("new data:" , new_step[0])
            #     print("pred", np.mean(prediction[:, agent, :step, :], axis=0))
            elif method == "random":
                choice = random.randint(0, 19)
                new_step[agent, :, :] = prediction[choice, agent, :step, :]

            elif method== "control":
                if not goal[i]:

                    distances = np.linalg.norm(prediction[:, agent, :step, :] - targets[i], axis=-1)  # Shape: (20, step)
                    best_option = np.argmin(np.sum(distances, axis=1)) # Get the closest option for each time step
                    new_step[agent, :, :] = prediction[best_option, agent, :step, :]  # Assign closest option

                    if np.any(np.abs(new_step[agent, :, :] - targets[i]) <= 1):
                        goal[i] = True
                        iter_message += f"Agent {agent + 1} achieved goal.\n"
                else:
                    choice = random.randint(0, 19)
                    new_step[agent, :, :] = prediction[choice, agent, :step, :]

        goal_status.append(iter_message)
        simulated = np.concatenate((simulated, new_step), axis=1)

        new_trajectory = np.concatenate((sample[0], new_step), axis=1)  # N, T+step, 2
        sample = torch.from_numpy(new_trajectory[:, -5:, :]).unsqueeze(0)  # add batch

        # if iter <2:
        #     print("simulated" , simulated.shape, true_data.shape)
            # print("new_trajectory", new_trajectory[0])
            # print("new sample", sample[0][0])
        iter += 1


    centroids = np.mean(simulated ,axis=0)
    return simulated, centroids, centroids, goal_status






def visualize_simulation(raw_agents, target_xy, model, length, steps, method, test_loader, args, fps, output_path, test_simulation=False,
                         agent_number=None, steps_control = None, steps_uncontrol = None, agent_indexes = None, choosing_option_by = None):
    if test_simulation:
        # print("dataset exp", test_loader.dataset.__len__())
        simulated = test_loader.dataset[1500][0]  # N, T, 2
        # print("simulated out", simulated)
        for i in range(int(length // 2)):
            five_time_steps = test_loader.dataset[i + 1501][0]
            # if i < 2:
                # print("five_time_steps", five_time_steps)
            simulated = np.concatenate((simulated, five_time_steps), axis=1)
            centroids = simulated.mean(axis=0)  # T, 2
    else:
        if steps_control:
            simulated, centroids, centroids_new = simulate_separate_controlled_uncontrolled(target_xy, output_path, model, length, steps_control, steps_uncontrol, method, test_loader, args, agent_indexes, choosing_option_by)

        elif raw_agents:
            simulated, centroids, centroids_new, goal_status =simulate_raw_data_with_agents(model, args, target_xy, test_loader, agent_indexes, length, steps, method)

        else:
            simulated, centroids, centroids_new = simulate(target_xy, model, length, steps, method, test_loader, args,
                                                           agent_number)
            goal_status = None
        print(simulated.shape, centroids.shape, centroids_new.shape)
    N, T, _ = simulated.shape

    fig, ax = plt.subplots(figsize=(8, 6))
    x_min, x_max = simulated[:, :, 0].min() - 1, simulated[:, :, 0].max() + 1
    y_min, y_max = simulated[:, :, 1].min() - 1, simulated[:, :, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Simulation of agents")

    # Set colors depending on agent_indexes
    if agent_indexes is None:
        colors = ['blue'] * N  # All agents blue
    elif raw_agents:
        colors = ['red' if agent in agent_indexes else 'blue' for agent in range(N)]
    else:
        colors = ['red' if agent in agent_indexes else 'pink' for agent in range(N)]

    scatter = ax.scatter([], [], s=80)
    scatter2 = ax.scatter([], [], s=80, marker='x')
    if raw_agents:
        target_x = [t[0] for t in target_xy]
        target_y = [t[1] for t in target_xy]
        target_marker = ax.scatter(target_x, target_y, color='pink', s=100, marker='^', label='Targets')
        goal_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, color="red",
                            bbox=dict(facecolor="white", alpha=0.7))

    else:
        target_marker = ax.scatter([target_xy[0]], [target_xy[1]], color='pink', s=100, marker='^', label='Target')

    if agent_number is not None:
        scatter3 = ax.scatter([], [], s=80, marker='+', alpha=0.7)
        # colors = plt.cm.tab10(np.linspace(0, 1, N + 2))

    def init():
        scatter.set_offsets([])
        scatter2.set_offsets([])
        # target_marker.set_offsets([target_xy])
        if raw_agents:
            goal_text.set_text("")
            return scatter, scatter2, goal_text
        return scatter, scatter2
        # return (scatter,)

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

        if raw_agents:
            if goal_status[frame]:
                goal_text.set_text(goal_status[frame])
            else:
                goal_text.set_text("")

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