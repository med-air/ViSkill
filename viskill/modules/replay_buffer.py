import gzip
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from ..utils.general_utils import AttrDict


#-------------------------Hindsight Experience Replay-------------------------
class HerReplayBuffer:
    def __init__(self, env_params, buffer_size, batch_size, sampler, T=None):
        # TODO(tao): unwrap env_params
        self.env_params = env_params
        self.T = T if T is not None else env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.batch_size = batch_size

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sampler
        
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['achieved_goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['act']]),
                        'dones': np.empty([self.size, self.T, 1]),
                        }
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, dones = episode_batch.obs, episode_batch.ag, episode_batch.g, \
                                                    episode_batch.actions, episode_batch.dones
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)

        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.buffers['dones'][idxs] = dones
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func.sample_her_transitions(temp_buffers, self.batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
    
    def save(self, save_dir, episode):
        with gzip.open(os.path.join(save_dir, f"replay_buffer_ep{episode}.zip"), 'wb') as f:
            pickle.dump(self.buffers, f)
        np.save(os.path.join(save_dir, f'idx_size_ep{episode}'), np.array([self.current_size, self.n_transitions_stored]))

    def load(self, save_dir, episode):
        with gzip.open(os.path.join(save_dir, f"replay_buffer_ep{episode}.zip"), 'rb') as f:
            self.buffers = pickle.load(f)
        idx_size = np.load(os.path.join(save_dir, f"idx_size_ep{episode}.npy"))
        self.current_size, self.n_transitions_stored = int(idx_size[0]), int(idx_size[1])


class HerReplayBufferWithGT(HerReplayBuffer):
    def __init__(self, env_params, buffer_size, batch_size, sampler, T=None):
        # TODO(tao): unwrap env_params
        self.env_params = env_params
        self.T = T if T is not None else env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.batch_size = batch_size
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sampler
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['achieved_goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['act']]),
                        'dones': np.empty([self.size, self.T, 1]),
                        'gt_g': np.empty([self.size, self.T, self.env_params['goal']]),
                        }
        self.sample_keys = ['obs', 'ag', 'g', 'actions', 'dones']
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, dones, mb_gt_g = episode_batch.obs, episode_batch.ag, episode_batch.g, \
                                                    episode_batch.actions, episode_batch.dones, episode_batch.gt_g
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)

        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.buffers['dones'][idxs] = dones
        self.buffers['gt_g'][idxs] = mb_gt_g
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self):
        temp_buffers = {}
        for key in self.sample_keys:
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func.sample_her_transitions(temp_buffers, self.batch_size)
        return transitions


class HER_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        
        # to get the params to re-compute reward
        transitions['r'] = self.reward_func(transitions['ag_next'], transitions['g'], None)
        if len(transitions['r'].shape) == 1:
            transitions['r'] = np.expand_dims(transitions['r'], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions


class HER_sampler_seq(HER_sampler):
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T-1, size=batch_size)   # from T to T-1
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        next_actions = episode_batch['actions'][episode_idxs, t_samples + 1].copy()
        transitions['next_actions'] = next_actions
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = self.reward_func(transitions['ag_next'], transitions['g'], None)
        if len(transitions['r'].shape) == 1:
            transitions['r'] = np.expand_dims(transitions['r'], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
    

def get_buffer_sampler(env, cfg):
    if cfg.type == 'her':
        sampler = HER_sampler(
            replay_strategy=cfg.strategy,
            replay_k=cfg.k,
            reward_func=env.compute_reward,
        )
    elif cfg.type == 'her_seq':
        sampler = HER_sampler_seq(
            replay_strategy=cfg.strategy,
            replay_k=cfg.k,
            reward_func=env.compute_reward,
        )
    else:
        raise NotImplementedError
    return sampler


def get_hier_buffer_samplers(env, cfg):
    reward_funcs = env.get_reward_functions()
    samplers = AttrDict()
    if cfg.type == 'her':
        for subtask in env.subtasks:
            samplers.update({subtask: HER_sampler(
                replay_strategy=cfg.strategy,
                replay_k=cfg.k,
                reward_func=reward_funcs[subtask],
            )})
    elif cfg.type == 'her_seq':
        for subtask in env.subtasks:
            samplers.update({subtask: HER_sampler_seq(
                replay_strategy=cfg.strategy,
                replay_k=cfg.k,
                reward_func=reward_funcs[subtask],
            )})
    else:
        raise NotImplementedError
    return samplers


#-------------------------Rollout Replay Buffer-------------------------
class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, len_cond):
        self.capacity = capacity
        self.batch_size = batch_size
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 
        
        self.obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        # gt_action should not be accessed unless entering terminal subtask
        # TODO: expose arm number
        self.gt_actions = np.empty((capacity, action_shape+len_cond), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done, gt_action):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.gt_actions[self.idx], gt_action)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_rollouts(self, rollouts):
        for transition in rollouts:
            self.add(*transition)

    def sample(self, idxs=None, return_idxs=False):
        if idxs is None:
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
    
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        gt_actions = self.gt_actions[idxs]
        if return_idxs:
            return obses, actions, rewards, next_obses, dones, gt_actions, idxs
        else:
            return obses, actions, rewards, next_obses, dones, gt_actions

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        done = self.dones[idx]
        gt_action = self.gt_actions[idx]

        return obs, action, reward, next_obs, done, gt_action

    def __len__(self):
        return self.capacity 