#!/usr/bin/env python
"""
A3C
"""

import os
import datetime
import argparse
from torch.multiprocessing import Value, Process

from hyperparams import HyperParams
from train import train
from model import ActorCritic
from env import create_environment
from monitor import Monitor
from optim import SharedAdam


def main():
    """
    Train an A3C agent
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_timesteps',
        default=10000000,
        type=int,
        help="How many total timesteps to run between all environments"
    )
    parser.add_argument(
        '--batch_size',
        default=20,
        type=int,
        help="How many steps to do before reflecting on the batch"
    )
    parser.add_argument(
        '--env_name',
        default='PongNoFrameskip-v4',
        type=str,
        help="Which environment to train on"
    )
    parser.add_argument(
        '--discount_factor',
        default=0.99,
        type=float,
        help=("The disount factor, also called gamma, used for discounting "
              "future returns")
    )
    parser.add_argument(
        '--gae',
        default=1.,
        type=float,
        help="Parameter for use in GAE, also called tau"
    )
    parser.add_argument(
        '--actor_coef',
        default=1.,
        type=float,
        help="How much weight to give the actor when updating"
    )
    parser.add_argument(
        '--critic_coef',
        default=0.5,
        type=float,
        help="How much weight to give the critic when updating"
    )
    parser.add_argument(
        '--entropy_coef',
        default=0.01,
        type=float,
        help="How much weight to give entropy when updating"
    )
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help="Optimizer learning rate"
    )
    parser.add_argument(
        '--no_of_workers',
        default=10,
        type=int,
        help="Number of parallel processes to run"
    )
    parser.add_argument(
        '--feature_type',
        default='cnn',
        type=str,
        help="""The feature extractor to use on the network input.
        Options are: cnn, mlp"""
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    hyperparams = HyperParams(
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        discount_factor=args.discount_factor,
        gae=args.gae,
        actor_coef=args.actor_coef,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        env_name=args.env_name,
        learning_rate=args.learning_rate,
        no_of_workers=args.no_of_workers,
        feature_type=args.feature_type
    )

    # Make temporary directory for logging
    directory = './runs/{}'.format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Shared model
    atari = True if hyperparams.feature_type == 'cnn' else False
    temp_env = create_environment(args.env_name, monitor=False, atari=atari)
    shared_model = ActorCritic(temp_env.observation_space.shape,
                               temp_env.action_space.n,
                               hyperparams.feature_type)
    shared_model.share_memory()

    # Frame counter
    frame_counter = Value('i')

    # Optimizer
    optimizer = SharedAdam(shared_model.parameters(),
                           lr=hyperparams.learning_rate)
    optimizer.share_memory()

    # Monitor
    monitor = Monitor(directory)

    processes = []
    monitor_process = Process(target=monitor.monitor, args=(
        frame_counter,
        hyperparams.max_timesteps))
    monitor_process.start()
    processes.append(monitor_process)
    # for i in range(hyperparams.no_of_workers):
    #     process = Process(target=train, args=(
    #         shared_model,
    #         directory,
    #         hyperparams,
    #         frame_counter,
    #         optimizer,
    #         monitor.queue,
    #         i))
    #     process.start()
    #     processes.append(process)

    train(
        shared_model=shared_model,
        directory=directory,
        hyperparams=hyperparams,
        frame_counter=frame_counter,
        optimizer=optimizer,
        monitor_queue=monitor.queue,
        process_number=0
    )

    for process in processes:
        process.join()


if __name__ == '__main__':
    main()
