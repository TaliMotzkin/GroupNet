import torch
import numpy as np
import random


class LossCompute:
    def __init__(self, netG, netD, netM, args):
        self.netG = netG
        self.netD = netD
        self.netM = netM
        self.device = args.device
        self.bce = torch.nn.BCELoss()
        self.l2_weight = args.l2_weight
        self.args = args

    def compute_generator_loss(self, prediction, H , data, mission , agent,  target, agents_future_steps):
        l2=[]
        for i in range(5):
            #agents - (B*N,T,2)
            pred_trajectories = self.netG(prediction, H , data, mission , agent,  target)
            agents_future_steps = agents_future_steps.view(self.args.batch_size, 8, 10, 2)
            my_agents_true_steps = agents_future_steps[:, agent, :, :]
            loss_l2 =  self.l2_loss(pred_trajectories,my_agents_true_steps,'')
            l2.append(loss_l2)


        #Helps avoid "mode collapse" by allowing multiple solutions.
        l2=torch.stack(l2,dim=-1)
        l2=torch.min(l2,dim=-1)[0]
        l2_loss_sum = l2.mean()

        # agents_future_steps = agents_future_steps.view(self.args.batch_size, 8, 10, 2)
        agents_future_steps_pred = agents_future_steps.clone()
        agents_future_steps_pred[:, agent, :, :] = pred_trajectories


        scores_fake = self.netD(prediction, H , data,agent,agents_future_steps_pred)
        discriminator_loss = self.gan_g_loss(scores_fake)

        mission = mission.view(-1, 1)
        col_fake=self.netM(data, agents_future_steps_pred, target, H)
        col_loss=self.gan_c_loss(col_fake,mission)

        return self.l2_weight * l2_loss_sum + discriminator_loss + col_loss, l2_loss_sum.item(), discriminator_loss.item(), col_loss.item()

    def compute_generator_loss_real(self, prediction, H , past, future_traj):
        l2=[]
        for i in range(5):
            #agents - (B*N,T,2)
            pred_trajectories = self.netG(prediction, H , past)
            future_traj = future_traj.view(pred_trajectories.shape[0], 10, 2)
            loss_l2 =  self.l2_loss(pred_trajectories,future_traj,'')
            l2.append(loss_l2)


        #Helps avoid "mode collapse" by allowing multiple solutions.
        l2=torch.stack(l2,dim=-1)
        l2=torch.min(l2,dim=-1)[0]
        l2_loss_sum = l2.mean()


        scores_fake = self.netD(prediction, H ,past, pred_trajectories)
        discriminator_loss = self.gan_g_loss(scores_fake)



        return self.l2_weight * l2_loss_sum + discriminator_loss , l2_loss_sum.item(), discriminator_loss.item()

    def compute_discriminator_loss(self,prediction, H , data, mission , agent,  target, agents_future_steps ):
        pred_trajectories = self.netG(prediction, H , data, mission , agent,  target)
        agents_future_steps = agents_future_steps.view(self.args.batch_size, 8, 10, 2)
        agents_future_steps_pred = agents_future_steps.clone()
        agents_future_steps_pred[:,agent,:,: ] = pred_trajectories
        scores_fake = self.netD(prediction, H , data,agent,agents_future_steps_pred)
        scores_real = self.netD(prediction, H , data,agent, agents_future_steps)
        # print("scores_fake", scores_fake.shape)
        # print("scores_real", scores_real)
        loss_real, loss_fake = self.gan_d_loss(scores_fake, scores_real)  # BCEloss
        return loss_real + loss_fake , loss_real.item(), loss_fake.item(), scores_fake, scores_real


    def compute_discriminator_loss_real(self,prediction, H , past ,future_traj):
        pred_trajectories = self.netG(prediction, H , past)

        scores_fake = self.netD(prediction, H ,past, pred_trajectories)
        future_traj = future_traj.view(future_traj.shape[0]* future_traj.shape[1], 10, 2)
        scores_real = self.netD(prediction, H , past, future_traj)
        # print("scores_fake", scores_fake.shape)
        # print("scores_real", scores_real)
        loss_real, loss_fake = self.gan_d_loss(scores_fake, scores_real)  # BCEloss
        return loss_real + loss_fake , loss_real.item(), loss_fake.item(), scores_fake, scores_real

    def compute_Mission_loss(self, past_traj, future_steps, target, mission, H):
        col_fake=self.netM(past_traj, future_steps, target, H) #traj, pre speed, curr speed -> agents, seq of 7, xy
        # print("col_fake", col_fake.shape, col_fake)
        mission = mission.view(-1, 1)  # Now shape is (64, 1)
        # print("mission", mission.shape, mission)
        col_loss=self.gan_c_loss(col_fake,mission)
        # print("col_loss", col_loss.shape, col_loss)
        return col_loss

    def l2_loss(self, pred_traj, pred_traj_gt, mode='mean'):
            loss = (pred_traj_gt - pred_traj) ** 2
            if mode == 'sum':
                return torch.sum(loss)
            elif mode == 'mean':
                return loss.sum(dim=2).mean(dim=1)
            elif mode == 'raw':
                return loss.sum(dim=2).sum(dim=1)
            else:
                return loss.sum(dim=2)

    def gan_g_loss(self, scores_fake):
        y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.0)
        #Unlike the discriminator's gan_d_loss, the generator wants scores_fake to be close to 1.0
        return self.bce(scores_fake, y_fake)

    def gan_d_loss(self, scores_fake, scores_real):
        #Randomizing labels makes the discriminator less confident, preventing it from overpowering the generator
        # #Prevent overfitting in the discriminato.
        #Introduce uncertainty for better generalization:
        y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.0)
        y_fake = torch.ones_like(scores_fake) * random.uniform(0, 0.3)
        # print(scores_real.min(), scores_real.max())
        # print(y_real.min(), y_real.max())
        loss_real = self.bce(scores_real, y_real)
        loss_fake = self.bce(scores_fake, y_fake)
        return loss_real, loss_fake

    def gan_c_loss(self,col_score,col_label):
        return self.bce(col_score, col_label)



