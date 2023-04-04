import torch
import torch.nn.functional as F

from ..utils.general_utils import AttrDict, prefix_dict
from .sc_sac import SkillChainingSAC


class SkillChainingAWAC(SkillChainingSAC):
    def __init__(
            self,
            env_params,
            agent_cfg,
            sl_agent
    ):
        super().__init__(env_params, agent_cfg, sl_agent)

        # AWAC parameters
        self.n_action_samples = agent_cfg.n_action_samples
        self.lam = agent_cfg.lam

    def update_critic(self, obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask):
        assert subtask != self.env_params['last_subtask'] 

        with torch.no_grad():
            next_subtask = self.env_params['next_subtasks'][subtask]
            if next_subtask != self.env_params['last_subtask']:
                dist = self.actor(next_obs, next_subtask)
                action_out = dist.rsample()
                target_V = self.critic_target.q(next_obs, action_out, next_subtask)

        if self.raw_env_reward and next_subtask == self.env_params['last_subtask']:
            target_Q = self.reward_scale * reward  + (self.discount * raw_reward)
        else:
            target_Q =  self.reward_scale * reward + (self.discount * target_V).detach()

        current_Q1, current_Q2 = self.critic(obs, action, subtask)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize critic loss
        self.critic_optimizer[subtask].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[subtask].step()      
        
        metrics = AttrDict(
            critic_target_q=target_Q.mean().item(),
            critic_q=current_Q1.mean().item(),
            critic_loss=critic_loss.item()
        )
        return prefix_dict(metrics, subtask + '_')

    def update_actor(self, obs, action, subtask):
        # compute log probability
        dist = self.actor(obs, subtask)
        log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        # compute exponential weight
        weights = self._compute_weights(obs, action, subtask)
        actor_loss = -(log_probs * weights).sum()

        self.actor_optimizer[subtask].zero_grad()
        actor_loss.backward()
        self.actor_optimizer[subtask].step()

        metrics = AttrDict(
            log_probs=log_probs.mean(),
            actor_loss=actor_loss.item()
        )
        return prefix_dict(metrics, subtask + '_')

    def update(self, replay_buffer, demo_buffer):
        metrics = AttrDict()

        for i in range(self.update_epoch):
            for subtask in self.env_params['middle_subtasks']:
                # sample from replay buffer 
                obs, action, reward, next_obs, done, sl_norm_next_obs, raw_reward = self.get_samples(replay_buffer, subtask)
                action = action / self.max_action

                # update critic and actor
                metrics.update(self.update_critic(obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask))
                metrics.update(self.update_actor(obs, action, subtask))

                # sample from replay buffer 
                obs, action, reward, next_obs, done, sl_norm_next_obs, raw_reward = self.get_samples(demo_buffer, subtask)
                action = action / self.max_action

                # update critic and actor
                metrics.update(self.update_critic(obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask))
                metrics.update(self.update_actor(obs, action, subtask))

                # update target critic and actor
                self.update_target()

        return metrics
    
    def _compute_weights(self, obs, act, subtask):
        with torch.no_grad():
            batch_size = obs.shape[0]

            # compute action-value
            q_values = self.critic.q(obs, act, subtask)
            
            # sample actions
            policy_actions = self.actor.sample_n(obs, subtask, self.n_action_samples)
            flat_actions = policy_actions.reshape(-1, self.dima)

            # repeat observation
            reshaped_obs = obs.view(batch_size, 1, *obs.shape[1:])
            reshaped_obs = reshaped_obs.expand(batch_size, self.n_action_samples, *obs.shape[1:])
            flat_obs = reshaped_obs.reshape(-1, *obs.shape[1:])

            # compute state-value
            flat_v_values = self.critic.q(flat_obs, flat_actions, subtask)
            reshaped_v_values = flat_v_values.view(obs.shape[0], -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute normalized weight
            adv_values = (q_values - v_values).view(-1)
            weights = F.softmax(adv_values / self.lam, dim=0).view(-1, 1)

        return weights * adv_values.numel()
