from gymnasium.envs.registration import register

__version__ = "0.1.0"

register(
    id="MemoryDecayExp_9x9-v0",
    entry_point="crci_mem.envs.memory_decay_exp_env:MemoryDecayExpEnv",
)
register(
    id="MemoryDecayMDP_9x9-v0",
    entry_point="crci_mem.envs.memory_mdp:MemoryDecayMDP",
)

_LAZY = {
    "make_belief_wrapper": ("crci_mem.user_model.agent_forgetful", "make_belief_wrapper"),
    "ForgetfulHH": ("crci_mem.user_model.agent_forgetful", "ForgetfulHH"),
    "BeliefWrapper": ("crci_mem.user_model.wrappers", "BeliefWrapper"),
    "PersistentBeliefWrapper": ("crci_mem.user_model.wrappers", "PersistentBeliefWrapper"),
    "NestedParticleFilter": ("crci_mem.inference.nested_particle_filter", "NestedParticleFilter"),
    "AssistantParticleFilter": ("crci_mem.inference.nested_particle_filter", "AssistantParticleFilter"),
    "InferencePipeline": ("crci_mem.pipeline.pipeline", "InferencePipeline"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
