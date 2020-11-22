from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch
import gym
import gym_activesearchrl

def experiment(variant):
    expl_env = NormalizedBoxEnv(gym.make('activesearchrl-v0'))
    eval_env = NormalizedBoxEnv(gym.make('activesearchrl-v0'))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    # policy = TanhGaussianPolicy(
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     hidden_sizes=[M, M],
    # )
    policy1 = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    policy2 = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )    
    w1 = torch.randn(1, requires_grad=True)
    w2 = torch.randn(1, requires_grad=True)

    p1 = torch.exp(w1)/ (torch.exp(w1) + torch.exp(w2))
    p2 = torch.exp(w2)/ (torch.exp(w1) + torch.exp(w2))

    policy = (p1 * policy1) + (p2 * policy2)

    # data = torch.load('/Users/conor/Documents/PHD_RESEARCH/ACTIVE_SEARCH_AS_RL/rlkit/data/tabular-active-search-k1/tabular_active_search_k1_2020_11_10_16_18_25_0000--s-0/params.pkl')
    # qf1 = data['trainer/qf1']
    # qf2 = data['trainer/qf2']
    # target_qf1 = data['trainer/target_qf1']
    # target_qf2 = data['trainer/target_qf2']
    # policy = data['trainer/policy']

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=64,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=30000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=2000,
            min_num_steps_before_training=2000,
            max_path_length=2000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-5,
            qf_lr=3E-5,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('tabular_active_search_k1_low_combo_if_0_01_coeff_gmm', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
