#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from sacred.observers import MongoObserver
from dae import ex
ex.observers.append(MongoObserver.create(db_name='binding',
                                         prefix='random_search'))
nr_runs_per_dataset = 100
datasets = {
    'bars': 12, 
    'corners': 5,
    'shapes': 3,
    'multi_mnist': 3,
    'mnist_shape': 2,
    'simple_superpos':2
}

for ds, k in datasets.items():
    for i in range(nr_runs_per_dataset):
        ex.run(config_updates={'dataset.name': ds, 'verbose': False, 'em.k': k},
               named_configs=['random_search'])

# Multi-Train Runs
for ds, k in datasets.items():
    if ds == "simple_superpos": continue
    for i in range(nr_runs_per_dataset):
        ex.run(config_updates={
            'dataset.name': ds, 
            'dataset.train_set': 'train_multi',
            'em.k': k,
            'em.e_step': 'max',
            'verbose': False}, named_configs=['random_search'])

# MSE-Likelihood Runs
ex.observers = [MongoObserver.create(db_name='binding', prefix='mse_likelihood')]
for ds, k in datasets.items():
    for i in range(nr_runs_per_dataset):
        ex.run(config_updates={
            'dataset.name': ds,
            'dataset.salt_n_pepper': 0.3,
            'network_spec': 'Fr250',
            'em.k': k,
            'verbose': False}, named_configs=['random_search'])

