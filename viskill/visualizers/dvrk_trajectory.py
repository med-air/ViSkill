import os
import torch
from ..trainers.sc_trainer import SkillChainingTrainer
from ..utils.rl_utils import RolloutStorage
from ..components.logger import logger, WandBLogger, Logger
from ..components.checkpointer import CheckpointHandler
from ..modules.sampler import DvrkSampler

WANDB_PROJECT_NAME = 'skill_chaining'
WANDB_ENTITY_NAME = 'thuang22'


class DvrkTrajectoryCollector(SkillChainingTrainer):
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        self._setup_logger()
        self._setup_sampler()   # Sampler
        self._setup_misc()      # MISC

    def _setup_agent(self):
        super()._setup_agent()
        self.device = torch.device(self.cfg.device)

        checkpt_dir = self.cfg.sc_checkpoint_dir
        # TODO(tao): expose loading metric and dir speicification
        sub_checkpt_dir = checkpt_dir + f'/model'
        CheckpointHandler.load_checkpoint(
            sub_checkpt_dir, self.agent, self.device, episode='best')

    def _setup_logger(self):
        exp_name = f"SC_{self.cfg.task}_{self.cfg.agent.sc_agent.name}_{self.cfg.agent.sl_agent.name}_seed{self.cfg.seed}_dvrk"
        if self.cfg.postfix is not None:
            exp_name =  exp_name + '_' + self.cfg.postfix
        self.wb = WandBLogger(exp_name=exp_name, project_name=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME, \
                path=self.work_dir, conf=self.cfg)
        self.logger = Logger(self.work_dir)
        self.termlog = logger

    def _setup_sampler(self):
        self.train_sampler = DvrkSampler(self.train_env, self.agent, self.env_params)
        self.eval_sampler = DvrkSampler(self.eval_env, self.agent, self.env_params)


    def _setup_misc(self):
        self.model_dir = self.work_dir / 'dvrk'
        self.model_dir.mkdir(exist_ok=True)
        for file in os.listdir(self.model_dir):
            os.remove(self.model_dir / file)

    def collect_trajectory(self):
        '''Eval agent.'''
        eval_rollout_storage = RolloutStorage()
        for _ in range(self.cfg.n_eval_episodes):
            episode, _, env_steps, dvrk_traj = self.eval_sampler.sample_episode(is_train=False, render=True)
            eval_rollout_storage.append(episode)
        rollout_status = eval_rollout_storage.rollout_stats()

        global_step = env_steps
        global_episode = self.cfg.n_eval_episodes
        self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=global_step)
        with self.logger.log_and_dump_ctx(global_step, ty='eval') as log:
            log('episode_sr', rollout_status.avg_success_rate)
            log('episode_reward', rollout_status.avg_reward)
            log('episode_length', env_steps)
            log('episode', global_episode)
            log('step', global_step)

        del eval_rollout_storage

        import csv
        with open(self.model_dir / 'dvrk_trajectory.csv', 'w') as file:
            writer = csv.writer(file)
            for row in dvrk_traj:
                writer.writerow(row)


        return rollout_status.avg_success_rate 