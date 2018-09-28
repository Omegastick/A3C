"""
methods for training an a3c agent
"""

import time
from typing import NamedTuple
import numpy as np
import torch
from torch.multiprocessing import Queue

from hyperparams import HyperParams
from agent import ActorCriticAgent
from env import create_environment
from model import ActorCritic
from monitor import EpisodeData


class TimestepInfo(NamedTuple):
    """
    info collected from timestep in the environment.
    """
    value: torch.Tensor
    log_prob: torch.Tensor
    reward: float
    entropy: torch.Tensor


def train(
        shared_model: torch.nn.Module,
        directory: str,
        hyperparams: HyperParams,
        frame_counter: torch.multiprocessing.Value,
        optimizer: torch.optim.Optimizer,
        monitor_queue: Queue,
        process_number: int):
    """
    trains an a3c agent on an openai gym environment.
    """
    torch.manual_seed(process_number)

    # make environment
    atari = True if hyperparams.feature_type == 'cnn' else False
    monitor = process_number == 0
    env = create_environment(hyperparams.env_name, directory, atari=atari,
                             monitor=monitor)
    env.seed(process_number)
    state = env.reset()
    state = torch.from_numpy(state)
    done = False
    episode_reward = 0
    episode_length = 0
    episode_values = []
    episode_start_time = time.time()
    hidden_state = (torch.zeros(1, 256), torch.zeros(1, 256))

    # make agent
    model = ActorCritic(env.observation_space.shape, env.action_space.n,
                        hyperparams.feature_type)
    agent = ActorCriticAgent(model, shared_model)

    # training loop
    while frame_counter.value < hyperparams.max_timesteps:
        # load weights from shared model
        model.load_state_dict(shared_model.state_dict())

        # reset batch
        batch = []

        # run environment to get batch
        for _ in range(hyperparams.batch_size):
            action, value, log_prob, entropy, hidden_state = agent.act(
                state, hidden_state)

            state, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_length += 1
            episode_values.append(value.item())

            batch.append(TimestepInfo(
                value=value,
                log_prob=log_prob,
                reward=reward,
                entropy=entropy
            ))

            if done:
                state = env.reset()
                hidden_state = (torch.zeros(1, 256), torch.zeros(1, 256))

            state = torch.from_numpy(state)

            if done:
                now = time.time()
                episode_data = EpisodeData(
                    score=episode_reward,
                    length=episode_length,
                    average_value=np.mean(episode_values),
                    time_taken=now - episode_start_time
                )
                monitor_queue.put(episode_data)
                with frame_counter.get_lock():
                    frame_counter.value += episode_length
                episode_reward = 0
                episode_length = 0
                episode_values = []
                episode_start_time = now
                break

        # Get value of final timestep
        values = [x.value for x in batch]
        if done:
            values.append(torch.Tensor([0.]))
        else:
            _, value, _ = model(state, hidden_state)
            values.append(value)

        # reflect on batch
        critic_loss = 0
        actor_loss = 0
        gae = torch.Tensor([0])
        real_value = values[-1]

        # if -1 in [x.reward for x in batch]:
        #     import ipdb; ipdb.set_trace()

        for i in reversed(range(len(batch))):
            real_value = (hyperparams.discount_factor * real_value
                          + batch[i].reward)
            advantage = real_value - values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)

            value_delta = (batch[i].reward
                           + hyperparams.discount_factor
                           * values[i + 1].data
                           - values[i].data)
            gae = (gae
                   * hyperparams.discount_factor
                   * hyperparams.gae
                   + value_delta)

            actor_loss = (actor_loss
                          - batch[i].log_prob * torch.Tensor([gae])
                          - hyperparams.entropy_coef * batch[i].entropy)

        optimizer.zero_grad()

        loss = (critic_loss * hyperparams.critic_coef
                + actor_loss * hyperparams.actor_coef)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

        # Share gradients
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                break
            shared_param._grad = param.grad

        optimizer.step()

        hidden_state = (hidden_state[0].data, hidden_state[1].data)
