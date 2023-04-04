from .sc_awac import SkillChainingAWAC
from .sc_ddpg import SkillChainingDDPG
from .sc_sac import SkillChainingSAC
from .sc_sac_sil import SkillChainingSACSIL
from .sl_ddpgbc import SkillLearningDDPGBC
from .sl_dex import SkillLearningDEX

AGENTS = {
    'SL_DDPGBC': SkillLearningDDPGBC,
    'SL_DEX': SkillLearningDEX,
    'SC_DDPG': SkillChainingDDPG,
    'SC_AWAC': SkillChainingAWAC,
    'SC_SAC': SkillChainingSAC,
    'SC_SAC_SIL': SkillChainingSACSIL,
}

def make_sl_agent(env_params, sampler, cfg):
    if cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % cfg.name
    else:
        assert 'SL' in cfg.name 
        return AGENTS[cfg.name](
            env_params=env_params,
            sampler=sampler,
            agent_cfg=cfg
        )


def make_sc_agent(env_params, cfg, sl_agent):
    if cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % cfg.name
    else:
        assert 'SC' in cfg.name
        return AGENTS[cfg.name](
            env_params=env_params,
            agent_cfg=cfg,
            sl_agent=sl_agent
        )
    