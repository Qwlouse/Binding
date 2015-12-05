#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from dae import ex


@ex.named_config
def best_bars():
    dataset = {
        'name': 'bars',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.01
    }
    network_spec = "F64"
    net_filename = 'Networks/best_bars_dae.h5'

ex.run(named_configs=['best_bars'])


@ex.named_config
def best_corners():
    dataset = {
        'name': 'corners',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.01
    }
    network_spec = "F64"
    net_filename = 'Networks/best_corners_dae.h5'

ex.run(named_configs=['best_corners'])


@ex.named_config
def best_shapes():
    dataset = {
        'name': 'shapes',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.01
    }
    network_spec = "F64"
    net_filename = 'Networks/best_shapes_dae.h5'

ex.run(named_configs=['best_shapes'])


@ex.named_config
def best_simple_superpos():
    dataset = {
        'name': 'simple_superpos',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.01
    }
    network_spec = "F64"
    net_filename = 'Networks/best_simple_superpos_dae.h5'

ex.run(named_configs=['best_superpos'])


@ex.named_config
def best_multi_mnist():
    dataset = {
        'name': 'multi_mnist',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.01
    }
    network_spec = "F64"
    net_filename = 'Networks/best_multi_mnist_dae.h5'

ex.run(named_configs=['best_multi_mnist'])


@ex.named_config
def best_mnist_shape():
    dataset = {
        'name': 'mnist_shape',
        'salt_n_pepper': 0.5
    }
    training = {
        'learning_rate': 0.01
    }
    network_spec = "F64"
    net_filename = 'Networks/best_mnist_shape_dae.h5'

ex.run(named_configs=['best_mnist_shape'])
