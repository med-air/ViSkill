import copy

import torch
import torch.nn as nn

from ..modules.subnetworks import MLP


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.q = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q = self.q(sa)
        return q


class DoubleCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.q1 = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

        self.q2 = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return torch.min(q1, q2)


class SkillChainingCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, middle_subtasks, last_subtask):
        super().__init__()

        self.qs = nn.ModuleDict({
            subtask: Critic(
                in_dim=in_dim,
                hidden_dim=hidden_dim
            ) for subtask in middle_subtasks})
        self.qs.update({last_subtask: None})

    def __getitem__(self, key):
        return self.qs[key]

    def forward(self, state, action, subtask):
        q = self.qs[subtask](state, action)
        return q
    
    def init_last_subtask_q(self, last_subtask, critic):
        '''Initialize with pre-trained local q-function'''
        assert self.qs[last_subtask] is None
        self.qs[last_subtask] = copy.deepcopy(critic)


class SkillChainingDoubleCritic(SkillChainingCritic):
    def __init__(self, in_dim, hidden_dim, middle_subtasks, last_subtask):
        super(SkillChainingCritic, self).__init__()

        self.qs = nn.ModuleDict({
            subtask: DoubleCritic(
                in_dim=in_dim,
                hidden_dim=hidden_dim
            ) for subtask in middle_subtasks})
        self.qs.update({last_subtask: None})

    def forward(self, state, action, subtask):
        q1, q2 = self.qs[subtask](state, action)
        return q1, q2

    def q(self, state, action, subtask):
        return self.qs[subtask].q(state, action)
    