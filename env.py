"""
Helper methods for setting up an environment.
"""

from baselines.common import atari_wrappers
import gym
import numpy as np


def create_environment(
        env_name: str,
        run_directory: str = None,
        monitor: bool = True,
        atari: bool = True) -> gym.Env:
    """
    Create an environment and wrap it.
    """
    env = gym.make(env_name)
    if monitor:
        env = gym.wrappers.Monitor(env, run_directory, force=True,
                                   video_callable=lambda x: x % 10 == 0)
    if atari:
        if 'NoFrameskip' in env_name:
            env = atari_wrappers.MaxAndSkipEnv(env)
        env = atari_wrappers.wrap_deepmind(env, episode_life=False)
        env = AdaptiveNormalizedObservation(env)
        env = wrap_pytorch(env)

    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    https://github.com/higgsfield/RL-Adventure/blob/master/common/wrappers.py
    """

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_pytorch(env):
    """
    https://github.com/higgsfield/RL-Adventure/blob/master/common/wrappers.py
    """
    return ImageToPyTorch(env)


class AdaptiveNormalizedObservation(gym.ObservationWrapper):
    """
    Maintains a running mean and standard deviation of the observations from
    the environment and uses those to normalise those observations.
    Based on: https://github.com/ikostrikov/pytorch-a3c/blob/master/envs.py
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.running_mean = 0.
        self.running_std = 0.
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation: np.ndarray):
        self.num_steps += 1
        self.running_mean = (self.running_mean * self.alpha
                             + observation.mean() * (1 - self.alpha))
        self.running_std = (self.running_std * self.alpha
                            + observation.std() * (1 - self.alpha))

        unbiased_mean = self.running_mean / (1 - pow(self.alpha,
                                                     self.num_steps))
        unbiased_std = self.running_std / (1 - pow(self.alpha,
                                                   self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
