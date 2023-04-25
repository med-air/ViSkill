import os

import numpy as np

from .general_utils import AttrDict, RecursiveAverageMeter


def get_env_params(env, cfg):
    obs = env.reset()
    env_params = AttrDict(
        obs=obs['observation'].shape[0],
        achieved_goal=obs['achieved_goal'].shape[0],
        goal=obs['desired_goal'].shape[0],
        act=env.action_space.shape[0],
        act_rand_sampler=env.action_space.sample,
        max_timesteps=env.max_episode_steps,
        max_action=env.action_space.high[0],
    )
    if cfg.skill_chaining:
        env_params.update(AttrDict(
            act_sc=obs['achieved_goal'].shape[0] - env.len_cond, # withoug contact condition
            max_action_sc=env.max_action_range,
            adaptor_sc=env.goal_adapator,
            subtask_order=env.subtask_order,
            num_subtasks=len(env.subtask_order),
            subtask_steps=env.subtask_steps,
            subtasks=env.subtasks,
            next_subtasks=env.next_subtasks,
            prev_subtasks=env.prev_subtasks,
            middle_subtasks=env.next_subtasks.keys(),
            last_subtask=env.last_subtask,
            reward_funcs=env.get_reward_functions(),
            len_cond=env.len_cond
        ))
    return env_params


class ReplayCache:
    def __init__(self, T):
        self.T = T
        self.reset()

    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones = [], [], [], [], []

    def store_transition(self, obs, action, done):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)

    def store_obs(self, obs):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print(self.ag)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)

        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones)
        return episode


class ReplayCacheGT(ReplayCache):
    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones, self.gt_g = [], [], [], [], [], []

    def store_transition(self, obs, action, done, gt_goal):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)
        self.gt_g.append(gt_goal)

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print(self.ag)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)
        gt_g = np.expand_dims(np.array(self.gt_g.copy()), axis=0)

        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones, gt_g=gt_g)
        return episode
    
    
def init_demo_buffer(cfg, buffer, agent, subtask=None, update_normalizer=True):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = os.path.join(os.getcwd(),'surrol/data/demo')
    file_name = "data_"
    file_name += cfg.task
    file_name += "_" + 'random'
    if subtask is None:
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + cfg.subtask
    else:
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + subtask
    file_name += ".npz"

    demo_path = os.path.join(demo_path, file_name)
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs, demo_gt = demo['observations'], demo['actions'], demo['gt_actions']

    episode_cache = ReplayCacheGT(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])
        for i in range(buffer.T):
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i==(buffer.T-1),
                gt_goal=demo_gt[epsd][i]
            )
        episode = episode_cache.pop()
        buffer.store_episode(episode)
        if update_normalizer:
            agent.update_normalizer(episode)


def init_sc_buffer(cfg, buffer, agent, env_params):
    '''Load demonstrations into buffer and initilaize normalizer'''
    for subtask in env_params.subtasks:
        demo_path = os.path.join(os.getcwd(),'surrol/data/demo')
        file_name = "data_"
        file_name += cfg.task
        file_name += "_" + 'random'
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + subtask
        file_name += ".npz"

        demo_path = os.path.join(demo_path, file_name)
        demo = np.load(demo_path, allow_pickle=True)
        demo_obs, demo_acs, demo_gt = demo['observations'], demo['actions'], demo['gt_actions']

        for epsd in range(cfg.num_demo):
            obs = demo_obs[epsd][0]['observation']
            next_obs = demo_obs[epsd][-1]['observation']
            action = demo_obs[epsd][0]['desired_goal'][:-env_params.len_cond]
            # reward = sum([env_params.reward_funcs[subtask](demo_obs[epsd][i+1]['achieved_goal'], demo_obs[epsd][i+1]['desired_goal']) \
            #         for i in range(len(demo_acs[epsd]))])
            reward = env_params.reward_funcs[subtask](demo_obs[epsd][-1]['achieved_goal'], demo_obs[epsd][-1]['desired_goal'])
            #print(subtask, epsd, reward)
            done = subtask not in env_params.next_subtasks.keys()
            reward = done * reward
            gt_action = demo_gt[epsd][-1]
            buffer[subtask].add(obs, action, reward, next_obs, done, gt_action)
            if agent.sc_agent.normalize:
                # TODO: hide normalized
                agent.sc_agent.o_norm[subtask].update(obs) 


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""
    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        for rollout in self.rollouts:
            stats.update(AttrDict(
                avg_reward=np.stack(rollout.reward).sum(),
                avg_success_rate=rollout.success[-1],
            ))
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]