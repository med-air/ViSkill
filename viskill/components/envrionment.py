import abc
from contextlib import contextmanager

import gym
import numpy as np
import torch
from surrol.utils.pybullet_utils import (pairwise_collision,
                                         pairwise_link_collision)


def approx_collision(goal_a, goal_b, th=0.025):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1) < th


class SkillLearningWrapper(gym.Wrapper):
    def __init__(self, env, subtask, output_raw_obs):
        super().__init__(env)
        self.subtask = subtask
        self._start_subtask = subtask
        self._elapsed_steps = None
        self._output_raw_obs = output_raw_obs

    @abc.abstractmethod
    def _replace_goal_with_subgoal(self, obs):
        """Replace achieved goal and desired goal."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _subgoal(self):
        """Output goal of subtask."""
        raise NotImplementedError

    @contextmanager
    def switch_subtask(self, subtask=None):
        '''Temporally switch subtask, default: next subtask'''
        if subtask is not None:
            curr_subtask = self.subtask            
            self.subtask = subtask
            yield
            self.subtask = curr_subtask
        else:
            self.subtask = self.SUBTASK_PREV_SUBTASK[self.subtask]
            yield
            self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]


#-----------------------------BiPegTransfer-v0-----------------------------
class BiPegTransferSLWrapper(SkillLearningWrapper):
    '''Wrapper for skill learning'''
    SUBTASK_ORDER = {
        'grasp': 0,
        'handover': 1,
        'release': 2
    }    
    SUBTASK_STEPS = {
        'grasp': 45,
        'handover': 35,
        'release': 20
    }
    SUBTASK_RESET_INDEX = {
        'handover': 4,
        'release': 10
    }
    SUBTASK_RESET_MAX_STEPS = {
        'handover': 45,
        'release': 70
    }
    SUBTASK_PREV_SUBTASK = {
        'handover': 'grasp',
        'release': 'handover'
    }
    SUBTASK_NEXT_SUBTASK = {
        'grasp': 'handover',
        'handover': 'release'
    }
    SUBTASK_CONTACT_CONDITION = {
        'grasp': [0, 1],
        'handover': [1, 0],
        'release': [0, 0]
    }
    LAST_SUBTASK = 'release'
    def __init__(self, env, subtask='grasp', output_raw_obs=False):
        super().__init__(env, subtask, output_raw_obs)
        self.done_subtasks = {key: False for key in self.SUBTASK_STEPS.keys()}

    @property
    def max_episode_steps(self):
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps
        return self.SUBTASK_STEPS[self.subtask]

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        info['is_success'] = reward + 1
        done = self._elapsed_steps == self.max_episode_steps
        # save groud truth goal
        with self.switch_subtask(self.LAST_SUBTASK):
            info['gt_goal'] = self._replace_goal_with_subgoal(next_obs.copy())['desired_goal']

        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info,

    def reset(self):
        self.subtask = self._start_subtask
        if self.subtask not in self.SUBTASK_RESET_INDEX.keys():
            obs = self.env.reset() 
            self._elapsed_steps = 0 
        else:
            success = False
            while not success:
                obs = self.env.reset() 
                self.subtask = self._start_subtask
                self._elapsed_steps = 0 

                action, skill_index = self.env.get_oracle_action(obs)
                count, max_steps = 0, self.SUBTASK_RESET_MAX_STEPS[self.subtask]
                while skill_index < self.SUBTASK_RESET_INDEX[self.subtask] and count < max_steps:
                    obs, reward, done, info = self.env.step(action)
                    action, skill_index = self.env.get_oracle_action(obs)
                    count += 1

                # Reset again if failed
                with self.switch_subtask():
                    obs_ = self._replace_goal_with_subgoal(obs.copy())  # in case repeatedly replace goal
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)

    def _replace_goal_with_subgoal(self, obs):
        """Replace ag and g"""
        subgoal = self._subgoal()    
        psm1col = pairwise_collision(self.env.obj_id, self.psm1.body)
        psm2col = pairwise_collision(self.env.obj_id, self.psm2.body)

        if self.subtask == 'grasp':
            obs['achieved_goal'] = np.concatenate([obs['observation'][0: 3], obs['observation'][7: 10], [psm1col, psm2col]])
        elif self.subtask == 'handover':
            obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['observation'][0: 3], [psm1col, psm2col]])
        elif self.subtask == 'release':
            obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['achieved_goal'], [psm1col, psm2col]])
        obs['desired_goal'] = np.append(subgoal, self.SUBTASK_CONTACT_CONDITION[self.subtask])
        return obs

    def _subgoal(self):
        """Output goal of subtask"""
        goal = self.env.subgoals[self.SUBTASK_ORDER[self.subtask]]
        return goal

    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[-5:-2], g[-5:-2], None) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:-2], g[:-2], None) + 1
            contact_cond = np.all(ag[-2:]==g[-2:])
            reward = (goal_reach and contact_cond) - 1
        else:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[:,-5:-2], g[:,-5:-2], None).reshape(-1, 1) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:,:-2], g[:,:-2], None).reshape(-1, 1) + 1
            contact_cond = np.all(ag[:, -2:]==g[:, -2:], axis=1).reshape(-1, 1)
            reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
        return reward


class BiPegTransferSCWrapper(BiPegTransferSLWrapper):
    '''Wrapper for skill chaining.'''
    MAX_ACTION_RANGE = 4.
    REWARD_SCALE = 30.
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        info['step'] = 1 - reward
        done = self._elapsed_steps == self.SUBTASK_STEPS[self.subtask]
        reward = done * reward 
        info['subtask'] = self.subtask
        info['subtask_done'] = False
        info['subtask_is_success'] = reward 

        if done:
            info['subtask_done'] = True
            # Transit to next subtask (if current subtask is not terminal) and reset elapsed steps
            if self.subtask in self.SUBTASK_NEXT_SUBTASK.keys():
                done = False
                self._elapsed_steps = 0
                self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]
                info['is_success'] = False
                reward = 0
            else:
                info['is_success'] = reward 
            next_obs_ = self._replace_goal_with_subgoal(next_obs)

        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info

    def reset(self, subtask=None):
        self.subtask = self._start_subtask if subtask is None else subtask
        if self.subtask not in self.SUBTASK_RESET_INDEX.keys():
            obs = self.env.reset() 
            self._elapsed_steps = 0 
        else:
            success = False
            while not success:
                obs = self.env.reset() 
                self.subtask = self._start_subtask if subtask is None else subtask
                self._elapsed_steps = 0 

                action, skill_index = self.env.get_oracle_action(obs)
                count, max_steps = 0, self.SUBTASK_RESET_MAX_STEPS[self.subtask]
                while skill_index < self.SUBTASK_RESET_INDEX[self.subtask] and count < max_steps:
                    obs, reward, done, info = self.env.step(action)
                    action, skill_index = self.env.get_oracle_action(obs)
                    count += 1

                # Reset again if failed
                with self.switch_subtask():
                    obs_ = self._replace_goal_with_subgoal(obs.copy())  # in case repeatedly replace goal
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)

    #---------------------------Reward---------------------------
    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[-5:-2], g[-5:-2], None) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:-2], g[:-2], None) + 1
            contact_cond = np.all(ag[-2:]==g[-2:])
            reward = (goal_reach and contact_cond) - 1
        else:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[:,-5:-2], g[:,-5:-2], None).reshape(-1, 1) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:,:-2], g[:,:-2], None).reshape(-1, 1) + 1
            if self.subtask == 'grasp':
                raise NotImplementedError
            contact_cond = np.all(ag[:, -2:]==g[:, -2:], axis=1).reshape(-1, 1)
            reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
        return reward + 1

    def goal_adapator(self, goal, subtask, device=None):
        '''Make predicted goal compatible with wrapper'''
        if isinstance(goal, np.ndarray):
            return np.append(goal, self.SUBTASK_CONTACT_CONDITION[subtask])
        elif isinstance(goal, torch.Tensor):
            assert device is not None
            ct_cond = torch.tensor(self.SUBTASK_CONTACT_CONDITION[subtask], dtype=torch.float32)
            ct_cond = ct_cond.repeat(goal.shape[0], 1).to(device)
            adp_goal = torch.cat([goal, ct_cond], 1)
            return adp_goal

    def get_reward_functions(self):
        reward_funcs = {}
        for subtask in self.subtask_order.keys():
            with self.switch_subtask(subtask):
                reward_funcs[subtask] = self.compute_reward
        return reward_funcs

    @property
    def start_subtask(self):
        return self._start_subtask

    @property
    def max_episode_steps(self):
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps
        return self.env._max_episode_steps

    @property
    def max_action_range(self):
        return self.MAX_ACTION_RANGE

    @property
    def subtask_order(self):
        return self.SUBTASK_ORDER
    
    @property
    def subtask_steps(self):
        return self.SUBTASK_STEPS
    
    @property
    def subtasks(self):
        subtasks = []
        for subtask, order in self.subtask_order.items():
            if order >= self.subtask_order[self.start_subtask]:
                subtasks.append(subtask)
        return subtasks

    @property
    def prev_subtasks(self):
        return self.SUBTASK_PREV_SUBTASK 
    
    @property
    def next_subtasks(self):
        return self.SUBTASK_NEXT_SUBTASK 
    
    @property
    def last_subtask(self):
        return self.LAST_SUBTASK

    @property
    def len_cond(self):
        return len(self.SUBTASK_CONTACT_CONDITION[self.last_subtask])


class BiPegBoardSLWrapper(BiPegTransferSLWrapper):
    '''Wrapper for skill learning'''
    SUBTASK_STEPS = {
        'grasp': 30,
        'handover': 35,
        'release': 35
    }
    SUBTASK_RESET_INDEX = {
        'handover': 5,
        'release': 9
    }
    SUBTASK_RESET_MAX_STEPS = {
        'handover': 30,
        'release': 60
    }


class BiPegBoardSCWrapper(BiPegTransferSCWrapper, BiPegBoardSLWrapper):
    '''Wrapper for skill chaining'''
    SUBTASK_STEPS = {
        'grasp': 30,
        'handover': 35,
        'release': 35
    }
    SUBTASK_RESET_INDEX = {
        'handover': 5,
        'release': 9
    }
    SUBTASK_RESET_MAX_STEPS = {
        'handover': 30,
        'release': 60
    }


class MatchBoardSLWrapper(BiPegTransferSLWrapper, SkillLearningWrapper):
    '''Wrapper for skill learning'''
    SUBTASK_ORDER = {
        'pull': 0,
        'grasp': 1,
        'release': 2,
        'push': 3
    }    
    SUBTASK_STEPS = {
        'pull': 50,
        'grasp': 30,
        'release': 20,
        'push': 50
    }
    SUBTASK_RESET_INDEX = {
        'grasp': 5,
        'release': 9,
        'push': 11,
    }
    SUBTASK_RESET_MAX_STEPS = {
        'grasp': 60,
        'release': 90,
        'push': 110
    }
    SUBTASK_PREV_SUBTASK = {
        'grasp': 'pull',
        'release': 'grasp',
        'push': 'release'
    }
    SUBTASK_NEXT_SUBTASK = {
        'pull': 'grasp',
        'grasp': 'release',
        'release': 'push'
    }
    SUBTASK_CONTACT_CONDITION = {
        'pull': [0],
        'grasp': [0],
        'release': [0],
        'push': [0]
    }
    LAST_SUBTASK = 'push'
    def __init__(self, env, subtask='grasp', output_raw_obs=False):
        super().__init__(env, subtask, output_raw_obs)
        self.col_with_lid = False

    def reset(self):
        self.col_with_lid = False
        return super().reset()

    def _replace_goal_with_subgoal(self, obs):
        """Replace ag and g"""
        subgoal = self._subgoal()    

        # collision condition
        if not self.col_with_lid:
            self.col_with_lid = pairwise_link_collision(self.env.psm1.body, 4, self.env.obj_ids['fixed'][-1], self.env.target_row * 6 + 3) != () 

        if self.subtask in ['pull', 'push']:
            obs['achieved_goal'] = np.concatenate([obs['observation'][0: 3], obs['achieved_goal'][3:6], [int(self.col_with_lid)]])
            obs['desired_goal'] = np.concatenate([subgoal[0:3], subgoal[6:9], [0]])
        elif self.subtask in ['grasp', 'release']:
            obs['achieved_goal'] = np.concatenate([obs['observation'][0: 3], obs['achieved_goal'][:3], [int(self.col_with_lid)]])
            obs['desired_goal'] = np.concatenate([subgoal[0:3], subgoal[3:6], [0]])
        return obs

    #---------------------------Reward---------------------------
    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            goal_reach = self.env.compute_reward(ag[:-1], g[:-1], None) + 1
            reward = (goal_reach and (1 - ag[-1])) - 1
        else:
            goal_reach = self.env.compute_reward(ag[:, :-1], g[:, :-1], None).reshape(-1, 1) + 1
            reward = np.all(np.hstack([goal_reach, 1 - ag[:, -1].reshape(-1, 1)]), axis=1) - 1.
        return reward
        

class MatchBoardSCWrapper(MatchBoardSLWrapper, BiPegTransferSCWrapper):
    MAX_ACTION_RANGE = 4.
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        info['step'] = 1 - reward
        done = self._elapsed_steps == self.SUBTASK_STEPS[self.subtask]
        reward = done * reward 
        info['subtask'] = self.subtask
        info['subtask_done'] = False
        info['subtask_is_success'] = reward 
        
        if done:
            info['subtask_done'] = True
            # Transit to next subtask (if current subtask is not terminal) and reset elapsed steps
            if self.subtask in self.SUBTASK_NEXT_SUBTASK.keys():
                done = False
                self._elapsed_steps = 0
                self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]
                info['is_success'] = False
                reward = 0
            else:
                info['is_success'] = reward 
            next_obs_ = self._replace_goal_with_subgoal(next_obs)

        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info
    
    def reset(self, subtask=None):
        self.col_with_lid = False
        return BiPegTransferSCWrapper.reset(self, subtask)

    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            goal_reach = self.env.compute_reward(ag[:-1], g[:-1], None) + 1
            reward = (goal_reach and (1 - ag[-1])) - 1
        else:
            goal_reach = self.env.compute_reward(ag[:, :-1], g[:, :-1], None).reshape(-1, 1) + 1
            reward = np.all(np.hstack([goal_reach, 1 - ag[:, -1].reshape(-1, 1)]), axis=1) - 1.
        return reward + 1

    def goal_adapator(self, goal, subtask, device=None):
        '''Make predicted goal compatible with wrapper'''
        if isinstance(goal, np.ndarray):
            return np.append(goal, self.SUBTASK_CONTACT_CONDITION[subtask])
        elif isinstance(goal, torch.Tensor):
            assert device is not None
            ct_cond = torch.tensor(self.SUBTASK_CONTACT_CONDITION[subtask], dtype=torch.float32)
            ct_cond = ct_cond.repeat(goal.shape[0], 1).to(device)
            adp_goal = torch.cat([goal, ct_cond], 1)
            return adp_goal


#-----------------------------Make envrionment-----------------------------
def make_env(cfg):
    env = gym.make(cfg.task)
    if cfg.task == 'BiPegTransfer-v0':
        if cfg.skill_chaining:
            env = BiPegTransferSCWrapper(env, cfg.init_subtask, output_raw_obs=False)
        else:
            env = BiPegTransferSLWrapper(env, cfg.subtask, output_raw_obs=False)
    elif cfg.task == 'BiPegBoard-v0':
        if cfg.skill_chaining:
            env = BiPegBoardSCWrapper(env, cfg.init_subtask, output_raw_obs=False)
        else:
            env = BiPegBoardSLWrapper(env, cfg.subtask, output_raw_obs=False)
    elif cfg.task == 'MatchBoard-v0':
        if cfg.skill_chaining:
            env = MatchBoardSCWrapper(env, cfg.init_subtask, output_raw_obs=False)
        else:
            env = MatchBoardSLWrapper(env, cfg.subtask, output_raw_obs=False)
    else:
        raise NotImplementedError
    return env