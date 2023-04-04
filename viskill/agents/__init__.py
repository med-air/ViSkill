from .hier_agent import HierachicalAgent


def make_hier_agent(env_params, samplers, cfg):
    return HierachicalAgent(env_params, samplers, cfg)
