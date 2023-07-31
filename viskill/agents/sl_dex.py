import numpy as np
import torch
import torch.nn.functional as F

from ..utils.general_utils import AttrDict
from .sl_ddpgbc import SkillLearningDDPGBC


class SkillLearningDEX(SkillLearningDDPGBC):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.k = 5

    def get_samples(self, replay_buffer):
        '''Addtionally sample next action for guidance propagation'''
        transitions = replay_buffer.sample()

        # preprocess
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        obs = self.to_torch(inputs_norm)
        next_obs = self.to_torch(inputs_next_norm)
        action = self.to_torch(transitions['actions'])
        next_action = self.to_torch(transitions['next_actions'])
        reward = self.to_torch(transitions['r'])
        done = self.to_torch(transitions['dones'])
        return obs, action, reward, done, next_obs, next_action

    def update_critic(self, obs, action, reward, next_obs, next_obs_demo, next_action_demo):
        with torch.no_grad():
            next_action_out = self.actor_target(next_obs)
            target_V = self.critic_target(next_obs, next_action_out)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

            # exploration guidance
            topk_actions = self.compute_propagated_actions(next_obs, next_obs_demo, next_action_demo)
            act_dist = self.norm_dist(topk_actions, next_action_out)
            target_Q += self.aux_weight * act_dist 

            clip_return = 5 / (1 - self.discount)
            target_Q = torch.clamp(target_Q, -clip_return, 0).detach()

        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        # optimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 
        
        metrics = AttrDict(
            critic_q=Q.mean().item(),
            critic_target_q=target_Q.mean().item(),
            critic_loss=critic_loss.item(),
            bacth_reward=reward.mean().item()
        )
        return metrics

    def update_actor(self, obs, obs_demo, action_demo):
        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)

        topk_actions = self.compute_propagated_actions(obs, obs_demo, action_demo)
        act_dist = self.norm_dist(action_out, topk_actions) 
        actor_loss = -(Q_out + self.aux_weight * act_dist).mean()
        actor_loss += action_out.pow(2).mean()

        # optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = AttrDict(
            actor_loss=actor_loss.item(),
            act_dist=act_dist.mean().item()
        )
        return metrics

    def update(self, replay_buffer, demo_buffer):
        for i in range(self.update_epoch):
            # sample from replay buffer 
            obs, action, reward, done, next_obs, next_action = self.get_samples(replay_buffer)
            obs_, action_, reward_, done_, next_obs_, next_action_ = self.get_samples(demo_buffer)

            with torch.no_grad():
                next_action_out = self.actor_target(next_obs)
                target_V = self.critic_target(next_obs, next_action_out)
                target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

                l2_pair = torch.cdist(next_obs, next_obs_)
                topk_value, topk_indices = l2_pair.topk(self.k, dim=1, largest=False)
                topk_weight = F.softmin(topk_value.sqrt(), dim=1)
                topk_actions = torch.ones_like(next_action_)

                for i in range(topk_actions.size(0)):
                    topk_actions[i] = torch.mm(topk_weight[i].unsqueeze(0), next_action_[topk_indices[i]]).squeeze(0)
                intr = self.norm_dist(topk_actions, next_action_out)
                target_Q += self.aux_weight * intr 
                next_action_out_ =self.actor_target(next_obs_)
                target_V_ = self.critic_target(next_obs_, next_action_out_)
                target_Q_ = self.reward_scale * reward_ + (self.discount * target_V_).detach()
                intr_ = self.norm_dist(next_action_, next_action_out_)
                target_Q_ += self.aux_weight * intr_ 

                clip_return = 5 / (1 - self.discount)
                target_Q = torch.clamp(target_Q, -clip_return, 0).detach()
                target_Q_ = torch.clamp(target_Q_, -clip_return, 0).detach()


            Q = self.critic(obs, action)
            Q_ = self.critic(obs_, action_)
            critic_loss = F.mse_loss(Q, target_Q) + F.mse_loss(Q_, target_Q_)

            # optimize critic loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()      

            action_out = self.actor(obs)
            action_out_ = self.actor(obs_)
            Q_out = self.critic(obs, action_out)
            Q_out_ = self.critic(obs_, action_out_)

            with torch.no_grad():
                l2_pair = torch.cdist(obs, obs_)
                topk_value, topk_indices = l2_pair.topk(self.k, dim=1, largest=False)
                topk_weight = F.softmin(topk_value.sqrt(), dim=1)
                topk_actions = torch.ones_like(action)
                
                for i in range(topk_actions.size(0)):
                    topk_actions[i] = torch.mm(topk_weight[i].unsqueeze(0), action_[topk_indices[i]]).squeeze(0)

            intr2 = self.norm_dist(action_out, topk_actions)
            intr3 = self.norm_dist(action_out_, action_)

            # Refer to https://arxiv.org/pdf/1709.10089.pdf
            actor_loss = - (Q_out + self.aux_weight * intr2).mean()
            actor_loss += -(Q_out_ + self.aux_weight * intr3).mean()

            actor_loss += action_out.pow(2).mean()
            actor_loss += action_out_.pow(2).mean()

            #actor_loss += self.action_l2 * action_out.pow(2).mean()

            # Optimize actor loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.update_target()

        metrics = AttrDict(
            batch_reward=reward.mean().item(),
            critic_q=Q.mean().item(),
            critic_q_=Q_.mean().item(),
            critic_target_q=target_Q.mean().item(),
            critic_loss=critic_loss.item(),
            actor_loss=actor_loss.item()
        )
        return metrics