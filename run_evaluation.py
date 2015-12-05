#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from dae import ex

for ds in ['bars', 'corners', 'shapes', 'multi_mnist', 'mnist_shape', 'simple_superpos']:
    for k in [2, 3, 5, 12]:
        ex.run_command('evaluate', config_updates={
            'dataset.name': ds,
            'net_filename': 'Networks/{}_best_dae.h5'.format(ds),
            'em.k':k,
            'em.dump_results': 'Results/{}_20_{}.pickle'.format(ds, k),
            'seed': 1337})

