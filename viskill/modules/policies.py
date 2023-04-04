import copy
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .subnetworks import MLP
from .distributions import SquashedNormal


LOG_STD_BOUNDS = (-5, 2)

class DeterministicActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1.):
        super().__init__()

        self.trunk = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim
        )
        self.max_action = max_action

    def forward(self, state):
        a = self.trunk(state)
        return self.max_action * torch.tanh(a)


class DiagGaussianActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1.):
        super().__init__()

        self.trunk = MLP(
            in_dim=in_dim,
            out_dim=2*out_dim,
            hidden_dim=hidden_dim
        )
        self.max_action = max_action

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = LOG_STD_BOUNDS
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
    
    def sample_n(self, obs, n_samples):
        return self.forward(obs).sample_n(n_samples)

# class DiagGaussianActor(nn.Module):
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         hidden_dim: int,
#         min_log_std: float = -20.0,
#         max_log_std: float = 2.0,
#         min_action: float = -1.0,
#         max_action: float = 1.0,
#     ):
#         super().__init__()
#         self._mlp = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#         )
#         self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
#         self._min_log_std = min_log_std
#         self._max_log_std = max_log_std
#         self._min_action = min_action
#         self._max_action = max_action

#     def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
#         mean = self._mlp(state)
#         log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
#         policy = torch.distributions.Normal(mean, log_std.exp())
#         return policy

#     def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
#         policy = self._get_policy(state)
#         log_prob = policy.log_prob(action).sum(-1, keepdim=True)
#         return log_prob

#     def forward(self, state: torch.Tensor):
#         policy = self._get_policy(state)
#         action = policy.rsample()
#         action.clamp_(self._min_action, self._max_action)
#         log_prob = policy.log_prob(action).sum(-1, keepdim=True)
#         return action, log_prob

#     def act(self, state, device):
#         state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
#         policy = self._get_policy(state_t)
#         if self._mlp.training:
#             action_t = policy.sample()
#         else:
#             action_t = policy.mean
#         action = action_t[0].cpu().numpy()
#         return action


class SkillChainingActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1., 
                middle_subtasks=None, last_subtask=None):
        super().__init__()

        self.actors = nn.ModuleDict({
            subtask: DeterministicActor(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                max_action=max_action
            ) for subtask in middle_subtasks})
        self.actors.update({last_subtask: None})

    def __getitem__(self, key):
        return self.actors[key]

    def forward(self, state, subtask):
        a = self.actors[subtask](state)
        return a
    
    def init_last_subtask_actor(self, last_subtask, actor):
        '''Initialize with pre-trained local actor'''
        assert self.actors[last_subtask] is None
        self.actors[last_subtask] = copy.deepcopy(actor)


# class SkillChainingDiagGaussianActor(SkillChainingActor):
#     def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1.,  
#                 middle_subtasks=None, last_subtask=None):
#         super(SkillChainingActor, self).__init__()

#         self.actors = nn.ModuleDict({
#             subtask: DiagGaussianActor(
#                 state_dim=in_dim,
#                 action_dim=out_dim,
#                 hidden_dim=hidden_dim,
#                 min_action=-max_action,
#                 max_action=max_action
#             ) for subtask in middle_subtasks})
#         self.actors.update({last_subtask: None})

    # def sample_n(self, obs, subtask, n_samples):
    #     return self.actors[subtask].sample_n(obs, n_samples)

    # def squash_action(self, dist, raw_action):
    #     squashed_action = torch.tanh(raw_action)
    #     jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    #     log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
    #     return squashed_action, log_prob


class SkillChainingDiagGaussianActor(SkillChainingActor):
    def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1.,  
                middle_subtasks=None, last_subtask=None):
        super(SkillChainingActor, self).__init__()

        self.actors = nn.ModuleDict({
            subtask: DiagGaussianActor(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                max_action=max_action
            ) for subtask in middle_subtasks})
        self.actors.update({last_subtask: None})

    def sample_n(self, obs, subtask, n_samples):
        return self.actors[subtask].sample_n(obs, n_samples)

    def squash_action(self, dist, raw_action):
        squashed_action = torch.tanh(raw_action)
        jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
        return squashed_action, log_prob
