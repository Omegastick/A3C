"""
Abstraction layer over an agent's neural network.
"""
from typing import Tuple
import torch
import torch.nn.functional as F

from model import ActorCritic


class ActorCriticAgent(object):
    """
    Abstraction layer over an agent's neural network
    """

    def __init__(self, model: ActorCritic, shared_model: ActorCritic):
        self.model = model
        self.shared_model = shared_model

        self.model.load_state_dict(shared_model.state_dict())

    def act(self,
            state: torch.Tensor,
            hidden_state: torch.Tensor) -> Tuple[int, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        """
        Get an action, and some relevant information about how the action was
        determined.
        Returns: action, value, relevant_log_prob, entropy, hidden_state
        """
        raw_probs, value, hidden_state = self.model(state, hidden_state)

        probs = F.softmax(raw_probs, dim=1)
        log_probs = F.log_softmax(raw_probs, dim=1)
        entropy = -(log_probs * probs).sum(1, keepdim=True)

        action = probs.multinomial(num_samples=1)
        relevant_log_prob = log_probs.gather(1, action)

        return action.item(), value, relevant_log_prob, entropy, hidden_state
