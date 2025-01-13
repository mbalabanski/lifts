from gymnasium.envs.registration import register

# from envs.base_env import LiftsEnv

register(
    id="lifts/QuadRotor-v0",
    entry_point="lifts.envs:LiftsEnv",
)