#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from dae import ex

for ds in ['bars', 'corners', 'shapes', 'multi_mnist', 'mnist_shape', 'simple_superpos']:
    for k in [2, 3, 5, 12]:
        ex.run_command('evaluate', config_updates={
            'dataset.name': ds,
            'net_filename': 'Networks/best_{}_dae.h5'.format(ds),
            'em.k': k,
            'em.nr_iters': 10,
            'em.dump_results': 'Results/{}_10_{}.pickle'.format(ds, k),
            'seed': 1337})

# Longer results for bars convergence plot
for k in [2, 3, 5, 12]:
    ex.run_command('evaluate', config_updates={
                   'dataset.name': 'bars',
                   'net_filename': 'Networks/best_{}_dae.h5'.format('bars'),
                   'em.k': k,
                   'em.nr_iters': 20,
                   'em.dump_results': 'Results/{}_20_{}.pickle'.format('bars', k),
                   'seed': 42})

# Results for multi-object trained networks
for ds in ['bars', 'corners', 'shapes', 'multi_mnist', 'mnist_shape']:
    for k in [2, 3, 5, 12]:
        ex.run_command('evaluate', config_updates={
            'dataset.name': ds,
            'net_filename': 'Networks/best_{}_dae_train_multi.h5'.format(ds),
            'em.k': k,
            'em.nr_iters': 10,
            'em.e_step': 'max',
            'em.dump_results': 'Results/{}_10_{}_train_multi.pickle'.format(ds, k),
            'seed': 23})


