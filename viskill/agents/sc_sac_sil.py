from ..utils.general_utils import AttrDict, prefix_dict
from .sc_awac import SkillChainingAWAC


class SkillChainingSACSIL(SkillChainingAWAC):
    def __init__(
            self,
            env_params,
            agent_cfg,
            sl_agent
    ):
        super().__init__(env_params, agent_cfg, sl_agent)
        self.policy_delay = agent_cfg.policy_delay

    def update_actor_and_alpha(self, obs, action, subtask, sil=False):
        if sil:
            #metrics = super(SkillChainingSACSIL, self).update_actor(obs, action, subtask)
            # compute log probability
            dist = self.actor(obs, subtask)
            log_probs = dist.log_prob(action).sum(-1, keepdim=True)
            # compute exponential weight
            weights = self._compute_weights(obs, action, subtask)
            actor_loss = -(log_probs * weights).sum()

            # optimize actor loss
            self.actor_optimizer[subtask].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[subtask].step()

            metrics = AttrDict(
                log_probs=log_probs.mean(),
                actor_loss=actor_loss.item()
            )
            metrics = prefix_dict(metrics, 'sil_')
        else:
            # compute log probability
            #metrics = super(SkillChainingAWAC, self).update_actor_and_alpha(obs, subtask)
            dist = self.actor(obs, subtask)
            action = dist.rsample()
            log_probs = dist.log_prob(action).sum(-1, keepdim=True)
            # compute state value
            actor_Q = self.critic.q(obs, action, subtask)
            actor_loss = (- actor_Q).mean() 

            # optimize actor loss
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

        for subtask in self.env_params['middle_subtasks']:
            for i in range(self.update_epoch):
                # sample from replay buffer 
                obs, action, reward, next_obs, done, sl_norm_next_obs, raw_reward = self.get_samples(replay_buffer, subtask)
                action = (action / self.max_action)

                # update critic and actor
                metrics.update(self.update_critic(obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask))
                if (i + 1) % self.policy_delay == 0:
                    metrics.update(self.update_actor_and_alpha(obs, action, subtask, sil=False))
                
                # sample from replay buffer 
                obs, action, reward, next_obs, done, sl_norm_next_obs, raw_reward = self.get_samples(demo_buffer, subtask)
                action = (action / self.max_action)

                # update actor
                metrics.update(self.update_critic(obs, action, reward, next_obs, sl_norm_next_obs, raw_reward, subtask))
                metrics.update(self.update_actor_and_alpha(obs, action, subtask, sil=True))

                # update target critic and actor
                self.update_target()

        return metrics
