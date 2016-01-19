#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from dae import ex

@ex.named_config
def best_bars():
    dataset = {
        'name': 'bars',
        'salt_n_pepper': 0.0
    }
    training = {
        'learning_rate': 0.768014586935404
    }
    seed = 459182787
    network_spec = "Fr100"
    net_filename = 'Networks/best_bars_dae.h5'

ex.run(named_configs=['best_bars'])


@ex.named_config
def best_corners():
    dataset = {
        'name': 'corners',
        'salt_n_pepper': 0.0
    }
    training = {
        'learning_rate': 0.0019199822609484764
    }
    seed = 158253144
    network_spec = "Fr100"
    net_filename = 'Networks/best_corners_dae.h5'

ex.run(named_configs=['best_corners'])


@ex.named_config
def best_shapes():
    dataset = {
        'name': 'shapes',
        'salt_n_pepper': 0.4
    }
    training = {
        'learning_rate': 0.08314720669724956
    }
    seed = 845841083
    network_spec = "Ft500"
    net_filename = 'Networks/best_shapes_dae.h5'

ex.run(named_configs=['best_shapes'])


@ex.named_config
def best_multi_mnist():
    dataset = {
        'name': 'multi_mnist',
        'salt_n_pepper': 0.6
    }
    training = {
        'learning_rate': 0.011361917579645924
    }
    seed = 498470020
    network_spec = "Fr1000"
    net_filename = 'Networks/best_multi_mnist_dae.h5'

ex.run(named_configs=['best_multi_mnist'])


@ex.named_config
def best_mnist_shape():
    dataset = {
        'name': 'mnist_shape',
        'salt_n_pepper': 0.6
    }
    training = {
        'learning_rate': 0.0316848152096582
    }
    seed = 166717815
    network_spec = "Fs250"
    net_filename = 'Networks/best_mnist_shape_dae.h5'

ex.run(named_configs=['best_mnist_shape'])


@ex.named_config
def best_simple_superpos():
    dataset = {
        'name': 'simple_superpos',
        'salt_n_pepper': 0.1
    }
    training = {
        'learning_rate': 0.36662702472680564
    }
    seed = 848588405
    network_spec = "Fr100"
    net_filename = 'Networks/best_simple_superpos_dae.h5'

ex.run(named_configs=['best_simple_superpos'])


@ex.named_config
def best_bars_train_multi():
    dataset = {
        'name': 'bars',
        'train_set': 'train_multi',
        'salt_n_pepper': 0.8
    }
    training = {
        'learning_rate': 0.01219213699462807
    }
    seed = 141786426
    network_spec = "Fs100"
    net_filename = 'Networks/best_bars_dae_train_multi.h5'

ex.run(named_configs=['best_bars_train_multi'])


@ex.named_config
def best_corners_train_multi():
    dataset = {
        'name': 'corners',
        'train_set': 'train_multi',
        'salt_n_pepper': 0.7
    }
    training = {
        'learning_rate': 0.02603487482829947
    }
    seed = 872544498
    network_spec = "Fr100"
    net_filename = 'Networks/best_corners_dae_train_multi.h5'

ex.run(named_configs=['best_corners_train_multi'])


@ex.named_config
def best_shapes_train_multi():
    dataset = {
        'name': 'shapes',
        'train_set': 'train_multi',
        'salt_n_pepper': 0.9
    }
    training = {
        'learning_rate': 0.049401835193689486
    }
    seed = 702200962
    network_spec = "Fs100"
    net_filename = 'Networks/best_shapes_dae_train_multi.h5'

ex.run(named_configs=['best_shapes_train_multi'])


@ex.named_config
def best_multi_mnist_train_multi():
    dataset = {
        'name': 'multi_mnist',
        'train_set': 'train_multi',
        'salt_n_pepper': 0.9
    }
    training = {
        'learning_rate': 0.001785591525476118
    }
    seed = 632224571
    network_spec = "Fs250"
    net_filename = 'Networks/best_multi_mnist_dae_train_multi.h5'

ex.run(named_configs=['best_multi_mnist_train_multi'])


@ex.named_config
def best_mnist_shape_train_multi():
    dataset = {
        'name': 'mnist_shape',
        'train_set': 'train_multi',
        'salt_n_pepper': 0.6
    }
    training = {
        'learning_rate': 0.033199614969711265
    }
    seed = 900543563
    network_spec = "Fr1000"
    net_filename = 'Networks/best_mnist_shape_dae_train_multi.h5'

ex.run(named_configs=['best_mnist_shape_train_multi'])


