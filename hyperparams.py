"""
Provides a class for storing training hyperparmeters.
"""
from typing import NamedTuple


class HyperParams(NamedTuple):
    """
    Hyperparameters.
    """
    max_timesteps: int
    batch_size: int
    discount_factor: float
    gae: float
    actor_coef: float
    critic_coef: float
    entropy_coef: float
    env_name: str
    learning_rate: float
    no_of_workers: int
    feature_type: str
