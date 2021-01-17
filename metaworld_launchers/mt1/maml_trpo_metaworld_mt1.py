#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import copy
import click
import metaworld
import torch

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

# yapf: enable
@click.command()
@click.option('--env-name', type=str)
@click.option('--seed', type=int, default=1)
@click.option('--epochs', type=int, default=1000)
@click.option('--rollouts_per_task', type=int, default=10)
@click.option('--meta_batch_size', type=int, default=20)
@click.option('--entropy_coefficient', default=5e-6, type=float)
@wrap_experiment(snapshot_mode='gap', name_parameters='passed', snapshot_gap=50)
def maml_trpo_metaworld_mt1(ctxt, env_name, seed, epochs, rollouts_per_task, meta_batch_size, entropy_coefficient):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        rollouts_per_task (int): Number of rollouts per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    set_seed(seed)

    mt1 = metaworld.MT1(env_name)
    tasks = MetaWorldTaskSampler(mt1, 'train', add_env_onehot=True)
    test_sampler = copy.deepcopy(tasks)

    env = tasks.sample(1)[0]()

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 256, 256),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=torch.tanh,
        min_std=0.5,
        max_std=1.5,
        std_mlp_type='share_mean_std'
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=[256, 256, 256],
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    meta_evaluator = MetaEvaluator(test_task_sampler=test_sampler,
                                   n_test_tasks=50,
                                   n_exploration_eps=2)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=meta_batch_size)

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    sampler=sampler,
                    task_sampler=tasks,
                    value_function=value_function,
                    meta_batch_size=meta_batch_size,
                    discount=0.995,
                    gae_lambda=1.,
                    inner_lr=0.05,
                    num_grad_updates=1,
                    meta_evaluator=meta_evaluator,
                    entropy_method='max',
                    policy_ent_coeff=entropy_coefficient,
                    stop_entropy_gradient=True,
                    center_adv=False,)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=rollouts_per_task * env.spec.max_episode_length)


maml_trpo_metaworld_mt1()
