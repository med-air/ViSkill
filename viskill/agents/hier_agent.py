import torch

from ..components.checkpointer import CheckpointHandler
from ..utils.general_utils import AttrDict, prefix_dict
from .base import BaseAgent
from .factory import make_sc_agent, make_sl_agent


class HierachicalAgent(BaseAgent):
    def __init__(
            self,
            env_params,
            samplers,
            cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.subtasks = env_params.subtasks
        self.goal_adaptor = env_params['adaptor_sc']
        self.subtasks_steps = env_params['subtask_steps']
        self.middle_subtasks = env_params['middle_subtasks']
        self.last_subtask = env_params['last_subtask']
        self.len_cond = env_params['len_cond']
        self.curr_subtask = None

        self.sl_agent = torch.nn.ModuleDict(
            {subtask: make_sl_agent(env_params, samplers[subtask], cfg.sl_agent) for subtask in self.subtasks})
        self._init_sl_agent()
        self.sc_agent = make_sc_agent(env_params, cfg.sc_agent, self.sl_agent)
        self._init_sc_agent()

    def _init_sl_agent(self):
        checkpt_dir = self.cfg.checkpoint_dir
        for subtask in self.subtasks:
            # TODO(tao): expose loading metric and dir speicification
            sub_checkpt_dir = checkpt_dir + f'/{subtask}/model'
            CheckpointHandler.load_checkpoint(
                sub_checkpt_dir, self.sl_agent[subtask], self.device, episode=self.cfg.ckpt_episode)

    def _init_sc_agent(self):
        self.sc_agent.init(self.last_subtask, self.sl_agent)

    def update(self, sc_buffer, sl_buffer, sc_demo_buffer=None, sl_demo_buffer=None):
        metrics = AttrDict()
        if sc_demo_buffer is None:
            sc_metrics = prefix_dict(self.sc_agent.update(sc_buffer), 'sc_')
        else:
            sc_metrics = prefix_dict(self.sc_agent.update(sc_buffer, sc_demo_buffer), 'sc_')
        metrics.update(sc_metrics)
        
        if self.cfg.agent.update_sl_agent:
            for subtask in self.subtasks:
                sl_metrics = prefix_dict(self.sl_agent[subtask].update(sl_buffer[subtask]), 'sl_' + subtask + '_')
                metrics.update(sl_metrics)
        return metrics

    def get_action(self, obs, subtask, noise=False):
        output = AttrDict()
        if self._perform_hl_step_now(subtask):
            # perform step with skill-chaining policy
            if subtask not in self.middle_subtasks:
                self._last_sc_action = obs['desired_goal'][:-self.len_cond]
            else:
                self._last_sc_action = self.sc_agent.get_action(obs['observation'], subtask, noise=noise)
            output.is_sc_step = True
            self.curr_subtask = subtask
        else:
            output.is_sc_step = False

        # perform step with skill-learning policy
        assert self._last_sc_action is not None
        self.goal_adaption(obs, subtask) 
        sl_action = self.sl_agent[subtask].get_action(obs, noise=False) 

        output.update(AttrDict(
            sc_action=self._last_sc_action,
            sl_action=sl_action
        ))
        return output

    def goal_adaption(self, obs, subtask):
        # Add contact condition to make goal compatible with surrol wrapper
        if subtask == 'release' and self.cfg.task == 'MatchBoard-v0':
            adpt_goal = self.goal_adaptor(self._last_sc_action, subtask)
            adpt_goal[3: 6] = obs['desired_goal'][3: 6].copy()
            obs['desired_goal'] = adpt_goal
        elif subtask in ['push', 'pull'] and self.cfg.task == 'MatchBoard-v0':
            adpt_goal = self.goal_adaptor(self._last_sc_action, subtask)
            adpt_goal[3] = obs['desired_goal'][3].copy()
            adpt_goal[5] = obs['desired_goal'][5].copy()
            obs['desired_goal'] = adpt_goal
        else:
            obs['desired_goal'] = self.goal_adaptor(self._last_sc_action, subtask)

    def _perform_hl_step_now(self, subtask):
        """Indicates whether the skill-chaining policy should be executed in the current time step."""
        return subtask != self.curr_subtask
    
    def sync_networks(self):
        self.sc_agent.sync_networks()