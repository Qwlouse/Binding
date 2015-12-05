#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from dae import ex

for ds in ['bars', 'corners', 'shapes', 'multi_mnist', 'mnist_shape', 'simple_superpos']:
    for k in [2, 3, 5, 12]:
        ex.run_command('evaluate', config_updates={
            'dataset.name': ds,
            'net_filename': 'Networks/{}_best_dae.h5'.format(ds),
            'em.k': k,
            'em.nr_iters': 10,
            'em.dump_results': 'Results/{}_10_{}.pickle'.format(ds, k),
            'seed': 1337})

# Longer results for bars convergence plot
for k in [2, 3, 5, 12]:
    ex.run_command('evaluate', config_updates={
                   'dataset.name': 'bars',
                   'net_filename': 'Networks/{}_best_dae.h5'.format('bars'),
                   'em.k': k,
                   'em.nr_iters': 20,
                   'em.dump_results': 'Results/{}_20_{}.pickle'.format('bars', k),
                   'seed': 42})
