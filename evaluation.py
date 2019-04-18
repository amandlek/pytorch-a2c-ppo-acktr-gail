import json
import argparse
import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, eval_envs=None):

    if eval_envs is None:
        eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                                  None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    print(eval_episode_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
    )
    cmd_args = parser.parse_args()

    # load config used during training
    if not torch.cuda.is_available():
        model_dict = torch.load(cmd_args.agent, map_location=lambda storage, loc: storage)
    else:
        model_dict = torch.load(cmd_args.agent)

    print("loading model with config")
    print(model_dict['config'])
    from addict import Dict
    args = Dict(json.loads(model_dict['config']))

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # make envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    # load policy
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    actor_critic.load_state_dict(model_dict['policy'])

    ob_rms = model_dict['ob_rms']

    # run evaluation
    evaluate(actor_critic=actor_critic, 
        ob_rms=ob_rms, 
        env_name=args.env_name, 
        seed=args.seed, 
        num_processes=1, 
        eval_log_dir=args.log_dir,
        device=device, 
        eval_envs=envs)




