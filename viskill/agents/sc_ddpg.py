import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..components.normalizer import Normalizer
from ..modules.critics import SkillChainingCritic
from ..modules.policies import SkillChainingActor
from ..utils.general_utils import AttrDict
from .base import BaseAgent


class SkillChainingDDPG(BaseAgent):
    def __init__(
            self,
            env_params,
            agent_cfg,
            sl_agent,
    ):
        super().__init__()

        self.sl_agent = sl_agent

        self.discount = agent_cfg.discount
        self.reward_scale = agent_cfg.reward_scale
        self.update_epoch = agent_cfg.update_epoch
        self.device = agent_cfg.device
        self.env_params = env_params

        self.random_eps = agent_cfg.random_eps
        self.noise_eps = agent_cfg.noise_eps
        self.soft_target_tau = agent_cfg.soft_target_tau

        self.normalize = agent_cfg.normalize
        self.clip_obs = agent_cfg.clip_obs
        self.norm_clip = agent_cfg.norm_clip
        self.norm_eps = agent_cfg.norm_eps
        self.intr_reward = agent_cfg.intr_reward
        self.raw_env_reward = agent_cfg.raw_env_reward

        self.dima = env_params['act_sc']   #
        self.dimo = env_params['obs']
        self.max_action = env_params['max_action_sc']
        self.goal_adapator = env_params['adaptor_sc']

        # TODO: normarlizer 
        self.o_norm = {subtask: Normalizer(
            size=self.dimo, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps) for subtask in env_params['middle_subtasks']}

        # build policy
        self.actor = SkillChainingActor(
            in_dim=self.dimo, 
            out_dim=self.dima, 
            hidden_dim=agent_cfg.hidden_dim, 
            max_action=self.max_action,
            middle_subtasks=env_params['middle_subtasks'],
            last_subtask=env_params['last_subtask']
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)

        self.critic = SkillChainingCritic(
            in_dim=self.dimo+self.dima, 
            hidden_dim=agent_cfg.hidden_dim, 
            middle_subtasks=env_params['middle_subtasks'],
            last_subtask=env_params['last_subtask']
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        # optimizer
        self.actor_optimizer = {subtask: torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr) for subtask in env_params['middle_subtasks']}
        self.critic_optimizer = {subtask: torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr) for subtask in env_params['middle_subtasks']}

    def init(self, task, sl_agent):
        '''Initialize the actor, critic and normalizers of last subtask.'''
        self.actor.init_last_subtask_actor(task, sl_agent[task].actor)
        self.actor_target.init_last_subtask_actor(task, sl_agent[task].actor_target)
        self.critic.init_last_subtask_q(task, sl_agent[task].critic)
        self.critic_target.init_last_subtask_q(task, sl_agent[task].critic_target)
        self.sl_normarlizer = {subtask: sl_agent[subtask]._preproc_inputs
                                for subtask in self.env_params['subtasks']}

    def get_samples(self, replay_buffer, subtask):
        next_subtask = self.env_params['next_subtasks'][subtask]

        obs, action, reward, next_obs, done, gt_action, idxs = replay_buffer[subtask].sample(return_idxs=True)
        sl_norm_next_obs = self.sl_normarlizer[next_subtask](next_obs, gt_action, dim=1)    # only for terminal subtask
        obs = self._preproc_obs(obs, subtask)
        next_obs = self._preproc_obs(next_obs, next_subtask)
        action = self.to_torch(action)
        reward = self.to_torch(reward)
        done = self.to_torch(done)
        gt_action = self.to_torch(gt_action)

        if next_subtask == self.env_params['last_subtask'] and self.raw_env_reward:
            assert len(replay_buffer[subtask]) == len(replay_buffer[next_subtask])
            _, _, raw_reward, _, _, _ = replay_buffer[next_subtask].sample(idxs=idxs)
            return obs, action, reward, next_obs, done, sl_norm_next_obs, self.to_torch(raw_reward)

        return obs, action, reward, next_obs, done, sl_norm_next_obs, None
    
    def get_action(self, state, subtask, noise=False):
        # random action at initial stage
        with torch.no_grad():
            input_tensor = self._preproc_obs(state, subtask)
            action = self.actor[subtask](input_tensor).cpu().data.numpy().flatten()
            # Gaussian noise
            if noise:
                action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
                    -self.max_action, self.max_action)
        return action

    def update_critic(self, obs, action, reward, next_obs):
        metrics = AttrDict()

        with torch.no_grad():
            action_out = self.actor_target(next_obs)
            target_V = self.critic_target(next_obs, action_out)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

            clip_return = 1 / (1 - self.discount)
            target_Q = torch.clamp(target_Q, -clip_return, 0).detach()

        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        # Optimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()      
        
        metrics = AttrDict(
            critic_target_q=target_Q.mean().item(),
            critic_q=Q.mean().item(),
            critic_loss=critic_loss.item()
        )
        return metrics

    def update_actor(self, obs, action, is_demo=False):
        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)
        actor_loss = -(Q_out).mean()

        # Optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = AttrDict(
            actor_loss=actor_loss.item()
        ) 
        return metrics

    def update(self, replay_buffer):
        metrics = AttrDict()

        for i in range(self.update_epoch):
            # Sample from replay buffer 
            obs, action, reward, next_obs, done = self.get_samples(replay_buffer)
            # Update critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor(obs, action))

        # Update target critic and actor
        self.update_target()
        return metrics

    def _preproc_obs(self, o, subtask):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        if self.normalize and subtask != self.env_params.last_subtask:
            o = self.o_norm[subtask].normalize(o)
        inputs = torch.tensor(o, dtype=torch.float32).to(self.device)
        return inputs
    
    def update_target(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

    def update_normalizer(self, rollouts, subtask):
        for transition in rollouts:
            obs, _, _, _, _, _ = transition
            # update
            self.o_norm[subtask].update(obs)
            # recompute the stats
            self.o_norm[subtask].recompute_stats()