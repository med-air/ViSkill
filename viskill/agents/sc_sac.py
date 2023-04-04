import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..components.normalizer import Normalizer
from ..modules.critics import SkillChainingDoubleCritic
from ..modules.policies import SkillChainingDiagGaussianActor
from ..utils.general_utils import AttrDict, prefix_dict
from .sc_ddpg import SkillChainingDDPG


class SkillChainingSAC(SkillChainingDDPG):
    def __init__(
            self,
            env_params,
            agent_cfg,
            sl_agent
    ):
        super(SkillChainingDDPG, self).__init__()

        self.sl_agent = sl_agent

        self.discount = agent_cfg.discount
        self.reward_scale = agent_cfg.reward_scale
        self.update_epoch = agent_cfg.update_epoch
        self.device = agent_cfg.device
        self.raw_env_reward = agent_cfg.raw_env_reward
        self.env_params = env_params

        # SAC parameters
        self.learnable_temperature = agent_cfg.learnable_temperature
        self.soft_target_tau = agent_cfg.soft_target_tau

        self.normalize = agent_cfg.normalize
        self.clip_obs = agent_cfg.clip_obs
        self.norm_clip = agent_cfg.norm_clip
        self.norm_eps = agent_cfg.norm_eps

        self.dima = env_params['act_sc']   
        self.dimo = env_params['obs']
        self.max_action = env_params['max_action_sc']
        self.goal_adapator = env_params['adaptor_sc']

        # normarlizer
        self.o_norm = Normalizer(
            size=self.dimo, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )

        # build policy
        self.actor = SkillChainingDiagGaussianActor(
            in_dim=self.dimo, 
            out_dim=self.dima, 
            hidden_dim=agent_cfg.hidden_dim, 
            max_action=self.max_action,
            middle_subtasks=env_params['middle_subtasks'],
            last_subtask=env_params['last_subtask']
        ).to(agent_cfg.device)

        self.critic = SkillChainingDoubleCritic(
            in_dim=self.dimo+self.dima, 
            hidden_dim=agent_cfg.hidden_dim, 
            middle_subtasks=env_params['middle_subtasks'],
            last_subtask=env_params['last_subtask']
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        # entropy term 
        if self.learnable_temperature:
            self.target_entropy = -self.dima
            self.log_alpha = {subtask : torch.tensor(
                np.log(agent_cfg.init_temperature)).to(self.device) for subtask in env_params['middle_subtasks']}
            for subtask in env_params['middle_subtasks']:
                self.log_alpha[subtask].requires_grad = True
        else:
            self.log_alpha = {subtask : torch.tensor(
                np.log(agent_cfg.init_temperature)).to(self.device) for subtask in env_params['middle_subtasks']}

        # optimizer
        self.actor_optimizer = {subtask: torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr) for subtask in env_params['middle_subtasks']}
        self.critic_optimizer = {subtask: torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr) for subtask in env_params['middle_subtasks']}
        self.temp_optimizer = {subtask: torch.optim.Adam(
            [self.log_alpha[subtask]], lr=agent_cfg.temp_lr) for subtask in env_params['middle_subtasks']}

    def init(self, task, sl_agent):
        '''Initialize the actor, critic and normalizers of last subtask.'''
        self.actor.init_last_subtask_actor(task, sl_agent[task].actor)
        self.critic.init_last_subtask_q(task, sl_agent[task].critic)
        self.critic_target.init_last_subtask_q(task, sl_agent[task].critic_target)
        self.sl_normarlizer = {subtask: sl_agent[subtask]._preproc_inputs
                                for subtask in self.env_params['subtasks']}

    def alpha(self, subtask):
        return self.log_alpha[subtask].exp()

    def get_action(self, state, subtask, noise=False):
        with torch.no_grad():
        #state = {key: self.to_torch(state[key].reshape([1, -1])) for key in state.keys()}  # unsqueeze
            input_tensor = self._preproc_obs(state, subtask)
            dist = self.actor(input_tensor, subtask)
            if noise:
                action = dist.sample()
            else:
                action = dist.mean

        return action.cpu().data.numpy().flatten() * self.max_action

    def get_q_value(self, state, action, subtask):
        with torch.no_grad():
            input_tensor = self._preproc_obs(state, subtask)
            action = self.to_torch(action)
            q_value = self.critic.q(input_tensor, action, subtask)

        return q_value

    def update_critic(self, obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask):
        assert subtask != self.env_params['last_subtask'] 

        with torch.no_grad():
            next_subtask = self.env_params['next_subtasks'][subtask]
            if next_subtask != self.env_params['last_subtask']:
                dist = self.actor(next_obs, next_subtask)
                action_out = dist.rsample()
                log_prob = dist.log_prob(action_out).sum(-1, keepdim=True)
                target_V = self.critic_target.q(next_obs, action_out, next_subtask)
                target_V = target_V - self.alpha(next_subtask).detach() * log_prob
            else:
                action_out = self.actor[next_subtask](sl_norm_next_obs)
                target_V = self.critic[next_subtask](sl_norm_next_obs, action_out).squeeze(0)

        if self.raw_env_reward and next_subtask == self.env_params['last_subtask']:
            target_Q = self.reward_scale * reward  + (self.discount * raw_reward)
        else:
            target_Q =  self.reward_scale * reward + (self.discount * target_V).detach()

        current_Q1, current_Q2 = self.critic(obs, action, subtask)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize critic loss
        self.critic_optimizer[subtask].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[subtask].step()      
        
        metrics = AttrDict(
            critic_target_q=target_Q.mean().item(),
            critic_q=current_Q1.mean().item(),
            critic_loss=critic_loss.item()
        )
        return prefix_dict(metrics, subtask + '_')

    def update_actor_and_alpha(self, obs, subtask):
        # compute log probability
        dist = self.actor(obs, subtask)
        action = dist.rsample()
        log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        # compute state value
        actor_Q = self.critic.q(obs, action, subtask)
        actor_loss = (self.alpha(subtask).detach() * log_probs - actor_Q).mean()

        # optimize actor loss
        self.actor_optimizer[subtask].zero_grad()
        actor_loss.backward()
        self.actor_optimizer[subtask].step()

        metrics = AttrDict(
            log_probs=log_probs.mean(),
            actor_loss=actor_loss.item()
        )

        # compute temp loss
        if self.learnable_temperature:
            temp_loss = (self.alpha(subtask) * (-log_probs - self.target_entropy).detach()).mean()
            self.temp_optimizer[subtask].zero_grad()
            temp_loss.backward()
            self.temp_optimizer[subtask].step()

            metrics.update(AttrDict(
                temp_loss=temp_loss.item(),
                temp=self.alpha(subtask)
            ))
        return prefix_dict(metrics, subtask + '_')

    def update(self, replay_buffer, demo_buffer):
        metrics = AttrDict()

        for i in range(self.update_epoch):
            for subtask in self.env_params['middle_subtasks']:
                # sample from replay buffer 
                obs, action, reward, next_obs, done, sl_norm_next_obs, raw_reward = self.get_samples(replay_buffer, subtask)
                action = action / self.max_action

                # update critic and actor
                metrics.update(self.update_critic( obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask))
                metrics.update(self.update_actor_and_alpha(obs, subtask))
                
                # update target critic and actor
                self.update_target()

        return metrics

    def update_target(self):
        # update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)
