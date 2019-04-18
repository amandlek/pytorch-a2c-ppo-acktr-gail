import copy
import glob
import os
import time
import datetime
import json
import argparse
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from addict import Dict

### TODO: try to remove use_proper_time_limits from RolloutStorage? On timeout in non-terminal state, pure reward is replaced by bootstrap state value ###

class PPOConfig(Dict):
    def __init__(self):

        self.algo = 'ppo'
        self.env_name = 'Hopper-v2' #'Hopper-v2' # environment to train on
        self.seed = 1 # random seed
        self.num_processes = 1 # number of CPU processes to use

        self.num_env_steps = 10e6 # total number of environment steps to train
        self.num_steps = 200 #2048 # env steps per epoch of training
        self.use_proper_time_limits = False #True # compute returns, taking time limits into account

        self.cuda = True # whether to use CUDA
        self.cuda_deterministic = False # sets CUDA determinism (potentially slow!)

        self.recurrent_policy = False # whether to use recurrent policy

        self.lr = 3e-4 # learning rate
        self.eps = 1e-5 # RMSprop optimizer epsilon
        self.alpha = 0.99 # RMSprop optimizer alpha
        self.use_linear_lr_decay = True # linear schedule on the learning rate

        self.gamma = 0.99 # discount factor
        self.use_gae = True # generalized advantage estimation
        self.gae_lambda = 0.95
        self.entropy_coef = 0.0 # entropy coefficient
        self.value_loss_coef = 0.5 # value loss coefficient
        self.max_grad_norm = 0.5 # max norm of gradients

        self.ppo_epoch = 10 # number of PPO epochs
        self.num_mini_batch = 32 # number of batches for PPO
        self.clip_param = 0.2 # PPO clip parameter

        self.log_interval = 1 #10 # log interval, 1 log per n updates
        # self.save_interval = 100 # save interval, 1 save per n updates
        self.eval_interval = None # eval interval, 1 eval per n updates
        self.log_dir = './training_logs'
        # self.save_dir = './trained_models/'

        self.experiment.base_path = "./experiments"
        self.experiment.output_dir = "./trained_models"
        self.experiment.name = "test_ppo"

    def dump(self, filename=None):
        """
        Dumps the config to a json.
        If filename is not None, dump to file.
        Returns a string.
        """
        json_string = json.dumps(self.to_dict(), indent=4)
        if filename is not None:
            f = open(filename, "w")
            f.write(json_string)
            f.close()
        return json_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
    )
    cmd_args = parser.parse_args()

    args = PPOConfig()
    args.experiment.name = cmd_args.name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # tensorboard logging
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    exp_dir = os.path.join(args.experiment.base_path, args.experiment.name, time_str)
    os.makedirs(exp_dir)
    writer = SummaryWriter(exp_dir)

    # directory for saving model parameters
    output_dir = os.path.join(args.experiment.output_dir, args.experiment.name, time_str)
    os.makedirs(output_dir)
    save_timestamp = time.time()

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    # if args.algo == 'a2c':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         alpha=args.alpha,
    #         max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'ppo':
    #     agent = algo.PPO(
    #         actor_critic,
    #         args.clip_param,
    #         args.ppo_epoch,
    #         args.num_mini_batch,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'acktr':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    # if args.gail:
    #     assert len(envs.observation_space.shape) == 1
    #     discr = gail.Discriminator(
    #         envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
    #         device)
    #     file_name = os.path.join(
    #         args.gail_experts_dir, "trajs_{}.pt".format(
    #             args.env_name.split('-')[0].lower()))

    #     gail_train_loader = torch.utils.data.DataLoader(
    #         gail.ExpertDataset(
    #             file_name, num_trajectories=4, subsample_frequency=20),
    #         batch_size=args.gail_batch_size,
    #         shuffle=True,
    #         drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    # training epochs
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    total_episodes = 0
    for j in range(num_updates):

        epoch_start = time.time()

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # collect training data for current epoch by rolling out
        for step in range(args.num_steps):

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    writer.add_scalar('Train/Rollouts', info['episode']['r'], total_episodes)
                    total_episodes += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()

        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(gail_train_loader, rollouts,
        #                      utils.get_vec_normalize(envs)._obfilt)

        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step], rollouts.actions[step], args.gamma,
        #             rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        writer.add_scalar('Train/Value Loss', value_loss, j)
        writer.add_scalar('Train/Action Loss', action_loss, j)

        rollouts.after_update()

        # # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass

        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))

        # save model every 30 minutes or for the last epoch
        if time.time() - save_timestamp > 1800 or j == num_updates - 1: # 1800
            torch.save({
                'policy': actor_critic.state_dict(),
                'ob_rms': getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                'config': args.dump()
                }, 
                os.path.join(output_dir, "model_epoch_{}.pth".format(j)))
            save_timestamp = time.time()


        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        # tensorboard stuff
        writer.add_scalar("Train/Epoch Time", time.time() - epoch_start, j)
        writer.file_writer.flush()

if __name__ == "__main__":
    main()
