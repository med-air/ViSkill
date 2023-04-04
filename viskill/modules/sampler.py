from ..utils.general_utils import AttrDict, listdict2dictlist
from ..utils.rl_utils import ReplayCache, ReplayCacheGT


class Sampler:
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, env, agent, max_episode_len):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCacheGT(max_episode_len)

    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        return self._agent.get_action(obs, noise=is_train)
    
    def sample_episode(self, is_train, render=False):
        """Samples one episode from the environment."""
        self.init()
        episode, done = [], False
        while not done and self._episode_step < self._max_episode_len:
            action = self.sample_action(self._obs, is_train)
            if action is None:
                break
            if render:
                render_obs = self._env.render('rgb_array')
            obs, reward, done, info = self._env.step(action)
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done, info['gt_goal'])
            if render:
                episode[-1].update(AttrDict(image=render_obs))

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        episode[-1].done = True     # make sure episode is marked as done at final time step
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self._episode_cache.store_obs(self._obs)

    def _reset_env(self):
        return self._env.reset()


class HierarchicalSampler(Sampler):
    """Collects experience batches by rolling out a hierarchical agent. Aggregates low-level batches into HL batch."""
    def __init__(self, env, agent, env_params):
        super().__init__(env, agent, env_params['max_timesteps'])

        self._env_params = env_params
        self._episode_cache = AttrDict(
            {subtask: ReplayCache(steps) for subtask, steps in env_params.subtask_steps.items()})
    
    def sample_episode(self, is_train, render=False):
        """Samples one episode from the environment."""
        self.init()
        sc_transitions = AttrDict({subtask: [] for subtask in self._env_params.subtasks})
        sc_succ_transitions = AttrDict({subtask: [] for subtask in self._env_params.subtasks})
        sc_episode, sl_episode, done, prev_subtask_succ = [], AttrDict(), False, AttrDict()
        while not done and self._episode_step < self._max_episode_len:
            agent_output = self.sample_action(self._obs, is_train, self._env.subtask)
            if self.last_sc_action is None:
                self._episode_cache[self._env.subtask].store_obs(self._obs)

            if render:
                render_obs = self._env.render('rgb_array')
            if agent_output.is_sc_step:
                self.last_sc_action = agent_output.sc_action
                self.reward_since_last_sc = 0

            obs, reward, done, info = self._env.step(agent_output.sl_action)
            self.reward_since_last_sc += reward
            if info['subtask_done']:
                if not done:
                    # store skill-chaining transition
                    sc_transitions[info['subtask']].append(
                        [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])

                    if info['subtask_is_success']:
                        sc_succ_transitions[info['subtask']].append(
                            [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])
                    else:
                        sc_succ_transitions[info['subtask']].append([None])

                    # middle subtask 
                    self._episode_cache[self._env.subtask].store_obs(obs)
                    self._episode_cache[self._env.prev_subtasks[self._env.subtask]].\
                        store_transition(obs, agent_output.sl_action, True)      
                    self.last_sc_obs = obs['observation']
                else:
                    # terminal subtask
                    sc_transitions[info['subtask']] = []
                    sc_transitions[info['subtask']].append(
                        [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])
                    if info['subtask_is_success']:
                        sc_succ_transitions[info['subtask']].append(
                            [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])
                    else:
                        sc_succ_transitions[info['subtask']].append([None])
                    self._episode_cache[self._env.subtask].store_transition(obs, agent_output.sl_action, True)
                prev_subtask_succ[self._env.subtask] = info['subtask_is_success']
            else:
                self._episode_cache[self._env.subtask].store_transition(obs, agent_output.sl_action, False)
            sc_episode.append(AttrDict(
                reward=reward, 
                success=info['is_success'], 
                info=info))
            if render:
                sc_episode[-1].update(AttrDict(image=render_obs))

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        assert self._episode_step == self._max_episode_len
        for subtask in self._env_params.subtasks:
            if subtask not in prev_subtask_succ.keys():
                sl_episode[subtask] = self._episode_cache[subtask].pop()
                continue
            if prev_subtask_succ[subtask]:
                sl_episode[subtask] = self._episode_cache[subtask].pop()
            else:
                self._episode_cache[subtask].pop()

        sc_episode = listdict2dictlist(sc_episode)
        sc_episode.update(AttrDict(
            sc_transitions=sc_transitions,
            sc_succ_transitions=sc_succ_transitions)
        )
        
        return sc_episode, sl_episode, self._episode_step

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self.last_sc_obs, self.last_sc_action = self._obs['observation'], None  # stores observation when last hl action was taken
        self.reward_since_last_sc = 0   # accumulates the reward since the last HL step for HL transition

    def sample_action(self, obs, is_train, subtask):
        return self._agent.get_action(obs, subtask, noise=is_train)