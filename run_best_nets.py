#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from dae import ex


@ex.named_config
def best_bars():
    dataset = {
        'name': 'bars',
        'salt_n_pepper': 0.1
    }
    training = {
        'learning_rate': 0.0788887811150988
    }
    seed = 915841892
    network_spec = "Ft1000"
    net_filename = 'Networks/best_bars_dae.h5'

ex.run(named_configs=['best_bars'])


@ex.named_config
def best_corners():
    dataset = {
        'name': 'corners',
        'salt_n_pepper': 0.2
    }
    training = {
        'learning_rate': 0.11338088391400022
    }
    seed = 820141270
    network_spec = "Fs500"
    net_filename = 'Networks/best_corners_dae.h5'

ex.run(named_configs=['best_corners'])


@ex.named_config
def best_shapes():
    dataset = {
        'name': 'shapes',
        'salt_n_pepper': 0.3
    }
    training = {
        'learning_rate': 0.06701641168189125
    }
    seed = 533867354
    network_spec = "Ft1000"
    net_filename = 'Networks/best_shapes_dae.h5'

ex.run(named_configs=['best_shapes'])


@ex.named_config
def best_multi_mnist():
    dataset = {
        'name': 'multi_mnist',
        'salt_n_pepper': 0.8
    }
    training = {
        'learning_rate': 0.004194304163542453
    }
    seed = 708383804
    network_spec = "Fr500"
    net_filename = 'Networks/best_multi_mnist_dae.h5'

ex.run(named_configs=['best_multi_mnist'])


@ex.named_config
def best_mnist_shape():
    dataset = {
        'name': 'mnist_shape',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.011252506237215505
    }
    seed = 17521051
    network_spec = "Fr250"
    net_filename = 'Networks/best_mnist_shape_dae.h5'

ex.run(named_configs=['best_mnist_shape'])


@ex.named_config
def best_simple_superpos():
    dataset = {
        'name': 'simple_superpos',
        'salt_n_pepper': 0.1
    }
    training = {
        'learning_rate': 0.20713360779738232
    }
    seed = 563758549
    network_spec = "Fr500"
    net_filename = 'Networks/best_simple_superpos_dae.h5'

ex.run(named_configs=['best_simple_superpos'])
