import numpy as np
import argparse
import os
import sys
import random
sys.path.append(os.getcwd())
import torch
from data.dataloader_fish import FISHDataset, seq_collate
from model.GroupNet_nba import GroupNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn import functional as F
from XGB.XGB import run_xgb
import pandas as pd
from Simulator import visualize_simulation

class Constant:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # DIFF = 6
    X_MIN = 0
    X_MAX = 60
    # X_MAX = 100
    Y_MIN = 0
    Y_MAX = 60
    # Y_MAX = 50
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    # X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    X_CENTER = 30
    # Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35
    Y_CENTER = 30
    MESSAGE = 'You can rerun the script and choose any event from 0 to '

def draw_result(future,past,mode='pre'):
    # b n t 2
    print('drawing...')
    trajs = np.concatenate((past,future), axis = 2)
    print("traj",trajs.shape)
    batch = trajs.shape[0]
    for idx in range(20):
        plt.clf()
        traj = trajs[idx]#per batch?
        # traj = traj*94/28
        actor_num = traj.shape[0]
        length = traj.shape[1]
        
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                        ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid

        colorteam1 = 'dodgerblue'
        colorteam2 = 'orangered'
        colorball = 'limegreen'
        colorteam1_pre = 'skyblue'
        colorteam2_pre = 'lightsalmon'
        colorball_pre = 'mediumspringgreen'
		
        for j in range(actor_num):
            if j < 3:
                color = colorteam1
                color_pre = colorteam1_pre
            elif j < 6:
                color = colorteam2
                color_pre = colorteam2_pre
            else:
                color_pre = colorball_pre
                color = colorball
            for i in range(length):
                points = [(traj[j,i,0],traj[j,i,1])]
                (x, y) = zip(*points)
                # plt.scatter(x, y, color=color,s=20,alpha=0.3+i*((1-0.3)/length))
                if i < 5:
                    plt.scatter(x, y, color=color_pre,s=20,alpha=1)
                else:
                    plt.scatter(x, y, color=color,s=20,alpha=1)

            for i in range(length-1):
                points = [(traj[j,i,0],traj[j,i,1]),(traj[j,i+1,0],traj[j,i+1,1])]
                (x, y) = zip(*points)
                # plt.plot(x, y, color=color,alpha=0.3+i*((1-0.3)/length),linewidth=2)
                if i < 4:
                    plt.plot(x, y, color=color_pre,alpha=0.5,linewidth=2)
                else:
                    plt.plot(x, y, color=color,alpha=1,linewidth=2)

        court = plt.imread("datasets/fish/tank2.png")
        # court = plt.imread("datasets/nba/court.png")

        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN],alpha=0.5)
        if mode == 'pre':
            plt.savefig('vis/fish_overlap_softmax/'+str(idx)+'pre.png')
        else:
            plt.savefig('vis/fish_overlap_softmax/'+str(idx)+'gt.png')
    print('ok')
    return 


def vis_result(test_loader, args):
    total_num_pred = 0
    all_num = 0
    iter = 0
    for data in test_loader:
        if iter == 0:
            future_traj = np.array(data['future_traj']) * args.traj_scale # B,N,T,2
            past_traj = np.array(data['past_traj']) * args.traj_scale

            print("future", len(future_traj))#10
            print("past", len(past_traj))#10? batch!

            with torch.no_grad():
                prediction, distributions = model.inference(data)
            prediction = prediction * args.traj_scale
            prediction = np.array(prediction.cpu()) #(20,BN, T,2)
            print("pred length", len(prediction))#20
            batch = future_traj.shape[0]#10
            print("batch size", batch)
            actor_num = future_traj.shape[1] #20
            print("actor num", actor_num)
            print("distributions", distributions)

            y = np.reshape(future_traj,(batch*actor_num,args.future_length, 2)) #200, 10, 2
            y = y[None].repeat(20,axis=0)
            error = np.mean(np.linalg.norm(y- prediction,axis=3),axis=2)
            print("error", error.shape)
            indices = np.argmin(error, axis = 0)
            best_guess = prediction[indices,np.arange(batch*actor_num)]
            print("best_guess_prev", best_guess.shape)
            best_guess = np.reshape(best_guess, (batch,actor_num, args.future_length, 2))
            print("best_guess", best_guess.min(), best_guess.max())
            gt = np.reshape(future_traj,(batch,actor_num,args.future_length, 2))
            previous_3D = np.reshape(past_traj, (batch, actor_num, args.past_length, 2))

            # previous_3D = np.reshape(previous_3D,(batch,actor_num,args.future_length, 2))

            draw_result(best_guess,previous_3D)
            draw_result(gt,previous_3D,mode='gt')
        else:
            break
        iter += 1
    return





def test_model_all(test_loader, args, simple_dist_plot =False, dist_plot=False, measure_res=False, plot_20 = False, XGB=False):
    total_num_pred = 0
    all_num = 0
    l2error_overall = 0
    l2error_dest = 0
    l2error_avg_04s = 0
    l2error_dest_04s = 0
    l2error_avg_08s = 0
    l2error_dest_08s = 0
    l2error_avg_12s = 0
    l2error_dest_12s = 0
    l2error_avg_16s = 0
    l2error_dest_16s = 0
    l2error_avg_20s = 0
    l2error_dest_20s = 0
    l2error_avg_24s = 0
    l2error_dest_24s = 0
    l2error_avg_28s = 0
    l2error_dest_28s = 0
    l2error_avg_32s = 0
    l2error_dest_32s = 0
    l2error_avg_36s = 0
    l2error_dest_36s = 0


    l2error_overall_base = 0
    l2error_dest_base = 0
    l2error_avg_04s_base = 0
    l2error_dest_04s_base = 0
    l2error_avg_08s_base = 0
    l2error_dest_08s_base = 0
    l2error_avg_12s_base = 0
    l2error_dest_12s_base = 0
    l2error_avg_16s_base = 0
    l2error_dest_16s_base = 0
    l2error_avg_20s_base = 0
    l2error_dest_20s_base = 0
    l2error_avg_24s_base = 0
    l2error_dest_24s_base = 0
    l2error_avg_28s_base= 0
    l2error_dest_28s_base = 0
    l2error_avg_32s_base = 0
    l2error_dest_32s_base = 0
    l2error_avg_36s_base = 0
    l2error_dest_36s_base = 0

    all_X = []
    all_y = []
    iteration = 0
    for data in test_loader:
        # print("data", data['future_traj'].shape)

        past_traj = np.array(data['past_traj']) * args.traj_scale
        last_5_steps = past_traj[:, :, -5:, :]
        avg_velocity = np.mean(np.diff(last_5_steps, axis=2), axis=2, keepdims=True)
        last_position = last_5_steps[:, :, -1:, :]
        baseline_prediction = np.concatenate(
            [last_position + i * avg_velocity for i in range(1, 11)], axis=2
        )

        # print(past_traj.shape)
        # past_traj_printing = past_traj.reshape(past_traj.shape[0]*past_traj.shape[1], 5, 2)
        # print("past", past_traj_printing[9])
        # if iteration ==0:
            # print(last_5_steps[0,0], "last_5_steps")
            # print("baseline_prediction", baseline_prediction[0,0])

        with torch.no_grad():
            prediction, distributions, H = model.inference(data)
            # print("H out", H.shape)
            # print("data ", data["past_traj"].shape)
            # print("future_traj ", data["future_traj"].shape)
            # print("prediction", prediction.shape)
        if XGB:
            future_traj = data['future_traj']
            print("future shape", future_traj.shape)
            print("prediction shape ", prediction.permute(1,0,2,3).shape)
            X, y = run_xgb(prediction.permute(1,0,2,3), future_traj)

            softmax_dist = F.softmax(distributions, dim=1)
            all_X.append(X)
            all_y.append(y)
            print("softmax_dist", softmax_dist[9, :])

        future_traj = np.array(data['future_traj']) * args.traj_scale  # B,N,T,2

        prediction = prediction * args.traj_scale
        prediction = np.array(prediction.cpu()) #(20,BN,T,2)
        batch = future_traj.shape[0]
        actor_num = future_traj.shape[1]

        BN = batch * actor_num
        baseline_prediction = baseline_prediction.reshape(BN, args.future_length, 2)
        baseline_prediction = baseline_prediction[None].repeat(20, axis=0)

        y = np.reshape(future_traj,(batch*actor_num,args.future_length, 2))
        y = y[None].repeat(20,axis=0)

        if plot_20:
            predictions_np = prediction
            plt.figure(figsize=(8, 6))

            for sample in range(20):
                trajectory = predictions_np[sample, 6, :, :]  # (10, 2)
                plt.plot(trajectory[:, 0], trajectory[:, 1], marker='', linestyle='-',
                         label=f"Sample {sample}" if sample < 5 else None)

            mean_traj = predictions_np[:, 6, :, :].mean(axis=0)
            plt.plot(future_traj[0, 6, :, 0], future_traj[0, 6, :, 1], marker='o', linestyle='-', color='red', label="Ground Truth")
            plt.plot(mean_traj[ :, 0], mean_traj[:, 1], marker='o', linestyle='-', color='blue',
                     label="Mean trajectory")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.title(f"20 Sampled Trajectories for Agent {6} at time step 6")
            plt.legend(loc="best", fontsize=8, ncol=2)
            plt.grid(True)
            plt.savefig("20_trajectories.png")
            plt.show()

        ''' measurment researching '''
        if measure_res:
            print(prediction.shape, "prediction") #20, 2*8, 10, 2
            print("y" , (y[:, :, :3, :]).shape) #20, 16, 3, 2
            norma = np.linalg.norm(y[:, :, :3, :] - prediction[:, :, :3, :], axis=3) #computes the length [3,4] = 5, xy length!
            print("norma", norma.shape) # 20, 16, 3 (one norma per each tr)
            averging_norma = np.mean(norma, axis =2) #finding average norma(error length) across all time steps, batch*agents 20,16
            print("averging_norma", averging_norma.shape) #20,16 - for each of the 20 samples, we have the norma of error of 16 agents (averaged across time)
            min_averaging = np.min(averging_norma, axis=0) #out of the 20 samples, for each agent - we are taking the minimal error
            print("min_averaging", min_averaging.shape) #16
            final = np.mean(min_averaging) # 1 values of averaging across al agents
            print("final", final*batch) #then multipyong by 2 batch
            print(y.shape, "target")

        ''' distribution researching '''
        if simple_dist_plot:
            print(prediction.shape, "prediction")
            agent_idx = 9
            time_idx = 0
            xy_samples = prediction[:, agent_idx, time_idx, :] #20,2
            print("xy_samples", xy_samples)

            plt.figure(figsize=(6, 6))
            plt.scatter(xy_samples[:, 0], xy_samples[:, 1], color='blue', marker='o',
                        label=f'Agent {agent_idx}, Time {time_idx}')
            plt.title(f'Distribution of (x,y) for Agent {agent_idx} at Time {time_idx} across 20 options')
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(xy_samples[:, 0], bins=10, color='green', alpha=0.7)
            plt.xlabel('x coordinate')
            plt.ylabel('Count')
            plt.title('Histogram of x coordinates')

            plt.subplot(1, 2, 2)
            plt.hist(xy_samples[:, 1], bins=10, color='orange', alpha=0.7)
            plt.xlabel('y coordinate')
            plt.ylabel('Count')
            plt.title('Histogram of y coordinates')

            plt.tight_layout()
            plt.show()
        if dist_plot:
            mean_predictions = np.mean(prediction, axis=0) #8, 10, 2
            dists = np.linalg.norm(future_traj- prediction,axis=3) # 20,8,10, of all xy coords
            print("dists", dists.shape) #20, 8, 10
            min_dists = np.min(dists, axis=0) #8, 10
            best_option_idx = np.argmin(dists, axis=0) #taking the minimal option for each agent and each time step 8, 10, so each agent can have 10 time steps from different options...

            std_predictions = np.std(prediction, axis=0) #std of all 20 options.
            stderr_predictions = std_predictions / np.sqrt(prediction.shape[0])
            print("best_option_idx", best_option_idx.shape)
            print("std_predictions", std_predictions.shape) #8, 10, 2 #var of an agent in each time step and coords

            cmap = plt.get_cmap("tab20")
            for t in range(prediction.shape[2]):
                plt.figure(figsize=(8, 6))
                for agent in range(prediction.shape[1]):
                    pred_xy = mean_predictions[agent, t, :]
                    b_idx = best_option_idx [agent, t]
                    best_xy = prediction[b_idx, agent, t, :]

                    err_x = stderr_predictions[agent, t, 0]
                    err_y = stderr_predictions[agent, t, 1]

                    color = cmap(agent % 20)

                    plt.scatter(pred_xy[0], pred_xy[1],
                                s=80,   color=color , label=f"Agent {agent}") #mean movement
                    plt.scatter(best_xy[0], best_xy[1],
                                s=50, color=color,marker= '*', alpha=0.6) #best movement #todo i think i need to choose best movement as an entire traj, not taking timesteps from different options

                    gt_xy = future_traj[0][agent, t, :]
                    plt.scatter(gt_xy[0], gt_xy[1],
                                s=100,
                                color=color,
                                marker='x',
                                linewidths=2) #actual movement
                    plt.hlines(y=pred_xy[1], xmin=pred_xy[0] - err_x, xmax=pred_xy[0] + err_x,
                               color=color, linestyles='-', linewidth=2)
                    plt.vlines(x=pred_xy[0], ymin=pred_xy[1] - err_y, ymax=pred_xy[1] + err_y,
                               color=color, linestyles='-', linewidth=2)


                plt.title(f"Time Step {t}")
                plt.xlabel("X coordinate")
                plt.ylabel("Y coordinate")
                plt.grid(True)

                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

                plt.tight_layout()
                plt.show()


        l2error_avg_04s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:1,:] - prediction[:,:,:1,:], axis = 3),axis=2),axis=0))*batch #012
        l2error_dest_04s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,0:1,:] - prediction[:,:,0:1,:], axis = 3),axis=2),axis=0))*batch#012
        l2error_avg_08s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:2,:] - prediction[:,:,:2,:], axis = 3),axis=2),axis=0))*batch #024
        l2error_dest_08s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,1:2,:] - prediction[:,:,1:2,:], axis = 3),axis=2),axis=0))*batch#024
        l2error_avg_12s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:3,:] - prediction[:,:,:3,:], axis = 3),axis=2),axis=0))*batch#0.036
        l2error_dest_12s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,2:3,:] - prediction[:,:,2:3,:], axis = 3),axis=2),axis=0))*batch#0.036
        l2error_avg_16s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:4,:] - prediction[:,:,:4,:], axis = 3),axis=2),axis=0))*batch#0.48
        l2error_dest_16s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,3:4,:] - prediction[:,:,3:4,:], axis = 3),axis=2),axis=0))*batch#0.48
        l2error_avg_20s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:5,:] - prediction[:,:,:5,:], axis = 3),axis=2),axis=0))*batch#1
        l2error_dest_20s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,4:5,:] - prediction[:,:,4:5,:], axis = 3),axis=2),axis=0))*batch#1
        l2error_avg_24s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:6,:] - prediction[:,:,:6,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_24s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,5:6,:] - prediction[:,:,5:6,:], axis = 3),axis=2),axis=0))*batch#1.12
        l2error_avg_28s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:7,:] - prediction[:,:,:7,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_28s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,6:7,:] - prediction[:,:,6:7,:], axis = 3),axis=2),axis=0))*batch#1.24
        l2error_avg_32s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:8,:] - prediction[:,:,:8,:], axis = 3),axis=2),axis=0))*batch#1.36
        l2error_dest_32s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,7:8,:] - prediction[:,:,7:8,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_36s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:9,:] - prediction[:,:,:9,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_36s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,8:9,:] - prediction[:,:,8:9,:], axis = 3),axis=2),axis=0))*batch #1.48
        l2error_overall += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:10,:] - prediction[:,:,:10,:], axis = 3),axis=2),axis=0))*batch#2~!
        l2error_dest += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,9:10,:] - prediction[:,:,9:10,:], axis = 3),axis=2),axis=0))*batch



        l2error_avg_04s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:1,:] - baseline_prediction[:,:,:1,:], axis = 3),axis=2),axis=0))*batch #012
        l2error_dest_04s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,0:1,:] - baseline_prediction[:,:,0:1,:], axis = 3),axis=2),axis=0))*batch#012
        l2error_avg_08s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:2,:] - baseline_prediction[:,:,:2,:], axis = 3),axis=2),axis=0))*batch #024
        l2error_dest_08s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,1:2,:] - baseline_prediction[:,:,1:2,:], axis = 3),axis=2),axis=0))*batch#024
        l2error_avg_12s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:3,:] - baseline_prediction[:,:,:3,:], axis = 3),axis=2),axis=0))*batch#0.036
        l2error_dest_12s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,2:3,:] - baseline_prediction[:,:,2:3,:], axis = 3),axis=2),axis=0))*batch#0.036
        l2error_avg_16s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:4,:] - baseline_prediction[:,:,:4,:], axis = 3),axis=2),axis=0))*batch#0.48
        l2error_dest_16s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,3:4,:] - baseline_prediction[:,:,3:4,:], axis = 3),axis=2),axis=0))*batch#0.48
        l2error_avg_20s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:5,:] - baseline_prediction[:,:,:5,:], axis = 3),axis=2),axis=0))*batch#1
        l2error_dest_20s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,4:5,:] - baseline_prediction[:,:,4:5,:], axis = 3),axis=2),axis=0))*batch#1
        l2error_avg_24s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:6,:] - baseline_prediction[:,:,:6,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_24s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,5:6,:] - baseline_prediction[:,:,5:6,:], axis = 3),axis=2),axis=0))*batch#1.12
        l2error_avg_28s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:7,:] - baseline_prediction[:,:,:7,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_28s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,6:7,:] - baseline_prediction[:,:,6:7,:], axis = 3),axis=2),axis=0))*batch#1.24
        l2error_avg_32s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:8,:] - baseline_prediction[:,:,:8,:], axis = 3),axis=2),axis=0))*batch#1.36
        l2error_dest_32s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,7:8,:] - baseline_prediction[:,:,7:8,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_36s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:9,:] - baseline_prediction[:,:,:9,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_36s_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,8:9,:] - baseline_prediction[:,:,8:9,:], axis = 3),axis=2),axis=0))*batch #1.48
        l2error_overall_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:10,:] - baseline_prediction[:,:,:10,:], axis = 3),axis=2),axis=0))*batch#2~!
        l2error_dest_base += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,9:10,:] - baseline_prediction[:,:,9:10,:], axis = 3),axis=2),axis=0))*batch


        all_num += batch
        iteration += 1


    if XGB:
        X_final = np.vstack(all_X)
        y_final = np.concatenate(all_y)

        df = pd.DataFrame(X_final)
        df['target'] = y_final

        # Save to CSV
        df.to_csv("xgb_training_data.csv", index=False)

    print(all_num)
    l2error_overall /= all_num
    l2error_dest /= all_num
    l2error_avg_04s /= all_num
    l2error_dest_04s /= all_num
    l2error_avg_08s /= all_num
    l2error_dest_08s /= all_num
    l2error_avg_12s /= all_num
    l2error_dest_12s /= all_num
    l2error_avg_16s /= all_num
    l2error_dest_16s /= all_num
    l2error_avg_20s /= all_num
    l2error_dest_20s /= all_num
    l2error_avg_24s /= all_num
    l2error_dest_24s /= all_num
    l2error_avg_28s /= all_num
    l2error_dest_28s /= all_num
    l2error_avg_32s /= all_num
    l2error_dest_32s /= all_num
    l2error_avg_36s /= all_num
    l2error_dest_36s /= all_num

    l2error_overall_base /= all_num
    l2error_dest_base  /= all_num
    l2error_avg_04s_base  /= all_num
    l2error_dest_04s_base  /= all_num
    l2error_avg_08s_base  /= all_num
    l2error_dest_08s_base /= all_num
    l2error_avg_12s_base  /= all_num
    l2error_dest_12s_base  /= all_num
    l2error_avg_16s_base  /= all_num
    l2error_dest_16s_base  /= all_num
    l2error_avg_20s_base  /= all_num
    l2error_dest_20s_base  /= all_num
    l2error_avg_24s_base /= all_num
    l2error_dest_24s_base  /= all_num
    l2error_avg_28s_base  /= all_num
    l2error_dest_28s_base /= all_num
    l2error_avg_32s_base  /= all_num
    l2error_dest_32s_base /= all_num
    l2error_avg_36s_base  /= all_num
    l2error_dest_36s_base /= all_num

    print('##################')
    print('ADE 1.0s:',(l2error_avg_08s+l2error_avg_12s)/2)
    print('ADE 2.0s:',l2error_avg_20s)
    print('ADE 3.0s:',(l2error_avg_32s+l2error_avg_28s)/2)
    print('ADE 4.0s:',l2error_overall)

    print('FDE 1.0s:',(l2error_dest_08s+l2error_dest_12s)/2)
    print('FDE 2.0s:',l2error_dest_20s)
    print('FDE 3.0s:',(l2error_dest_28s+l2error_dest_32s)/2)
    print('FDE 4.0s:',l2error_dest)
    print('##################')

    # print('##################')
    # print('ADE 0.5s:', (l2error_avg_08s + l2error_avg_12s) / 2)
    # print('ADE 1.0s:', l2error_avg_20s)
    # print('ADE 1.5s:', (l2error_avg_32s + l2error_avg_28s) / 2)
    # print('ADE 2.0s:', l2error_overall)
    #
    # print('FDE 0.5s:', (l2error_dest_08s + l2error_dest_12s) / 2)
    # print('FDE 1.0s:', l2error_dest_20s)
    # print('FDE 1.5s:', (l2error_dest_28s + l2error_dest_32s) / 2)
    # print('FDE 2.0s:', l2error_dest)


    print('Base ##################')

    print('ADE 1s:', (l2error_avg_08s_base + l2error_avg_12s_base) / 2)
    print('ADE 2s:', l2error_avg_20s_base)
    print('ADE 3s:', (l2error_avg_32s_base + l2error_avg_28s_base) / 2)
    print('ADE 4s:', l2error_overall_base)

    print('FDE 1s:', (l2error_dest_08s_base + l2error_dest_12s_base) / 2)
    print('FDE 2s:', l2error_dest_20s_base)
    print('FDE 3s:', (l2error_dest_28s_base+ l2error_dest_32s_base) / 2)
    print('FDE 4s:', l2error_dest_base)
    print('##################')

    print('######### discrepancies ##############')

    error_ADE_1 = (((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s+l2error_avg_12s)/2)
    percentage_ADE_1 = error_ADE_1/((l2error_avg_08s + l2error_avg_12s) / 2)
    print(f'ADE 1s: error:{error_ADE_1}, {percentage_ADE_1*100}_%')

    error_ADE_2 = l2error_avg_20s_base - l2error_avg_20s
    percentage_ADE_2 = error_ADE_2/l2error_avg_20s
    print(f'ADE 2s: error:{error_ADE_2}, {percentage_ADE_2 * 100}_%')

    error_ADE_3 = ((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s+l2error_avg_28s) / 2
    percentage_ADE_3 = error_ADE_3/((l2error_avg_32s+l2error_avg_28s)/2)
    print(f'ADE 3s: error:{error_ADE_3}, {percentage_ADE_3 * 100}_%')

    error_ADE_4 = l2error_overall_base - l2error_overall
    percentage_ADE_4 = error_ADE_4 / l2error_overall
    print(f'ADE 4s: error:{error_ADE_4}, {percentage_ADE_4 * 100}_%')


    error_FDE_1 = (((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s + l2error_dest_12s) / 2)
    percentage_FDE_1 = error_FDE_1 / ((l2error_dest_08s + l2error_dest_12s) / 2)
    print(f'FDE 1s: error:{error_FDE_1}, {percentage_FDE_1 * 100}_%')

    error_FDE_2 = l2error_dest_20s_base - l2error_dest_20s
    percentage_FDE_2 = error_FDE_2 / l2error_dest_20s
    print(f'FDE 2s: error:{error_FDE_2}, {percentage_FDE_2 * 100}_%')

    error_FDE_3 = (((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s + l2error_dest_32s) / 2)
    percentage_FDE_3 = error_FDE_3 / ((l2error_dest_28s + l2error_dest_32s) / 2)
    print(f'FDE 3s: error:{error_FDE_3}, {percentage_FDE_3 * 100}_%')

    error_FDE_4 = l2error_dest_base - l2error_dest
    percentage_FDE_4 = error_FDE_4 / l2error_dest
    print(f'FDE 2s: error:{error_FDE_4}, {percentage_FDE_4 * 100}_%')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model_names', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_save_dir', default='saved_models/fish_overlap')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--traj_scale', type=int, default=1)
    parser.add_argument('--sample_k', type=int, default=20)
    parser.add_argument('--past_length', type=int, default=5)
    parser.add_argument('--future_length', type=int, default=10)

    args = parser.parse_args()

    """ setup """
    names = [x for x in args.model_names.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)


    test_dset = FISHDataset(
        obs_len=args.past_length,
        pred_len=args.future_length,
        training=False)

    test_loader = DataLoader(
        test_dset,
        batch_size=128,
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
        saved_path = os.path.join(args.model_save_dir,str(name)+'.p')
        print('load model from:',saved_path)
        checkpoint = torch.load(saved_path, map_location='cpu')
        training_args = checkpoint['model_cfg']

        model = GroupNet(training_args,device)            
        model.set_device(device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)

        if args.vis:
            vis_result(test_loader, args)
        # test_model_all(test_loader, args)
        # test  = np.load('datasets/nba/test.npy')
        # print('test :', test.shape, test[:5])
        # simulate(60, 1,'mean', test_loader, args)

        # visualize_simulation(False, [80,90], model, 160, 1,'random', test_loader, args, 5, '3agents_5to1_steps_random_closest_centroid_sample1500.gif',
        #                   steps_control = 5, steps_uncontrol=1, agent_indexes=[0, 5, 6], choosing_option_by = "closest_centroid")

        # visualize_simulation(True, [[80,90], [20,30], [40,60],[30,50]], model, 160, 1,'control', test_loader,
        #                      args, 5,'raw_controled_4agent_1step_control_sample0.gif', agent_indexes=[0, 1, 2, 3])

        visualize_simulation(False, [80, 90], model, 300, 1, 'first', test_loader, args, 10,
                             'choosing_first_traj.gif')