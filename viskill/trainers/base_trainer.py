from pathlib import Path
from abc import abstractmethod

class BaseTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.work_dir = Path(cfg.cwd)
        self._setup()

    @abstractmethod
    def _setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        '''Training agent'''
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        '''Evaluating agent.'''
        raise NotImplementedError
