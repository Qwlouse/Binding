#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from sacred.observers import MongoObserver
from dae import ex
ex.observers.append(MongoObserver.create(db_name='binding',
                                         prefix='random_search'))
nr_runs_per_dataset = 100
datasets = ['bars', 'corners', 'shapes', 'multi_mnist', 'mnist_shape',
            'easy_superpos']

for ds in datasets:
    for i in range(nr_runs_per_dataset):
        ex.run(config_updates={'dataset.name': ds, 'verbose': False},
               named_configs=['random_search'])
