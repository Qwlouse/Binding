#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import os

import h5py
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

import brainstorm as bs
from brainstorm import optional as opt
from brainstorm.tools import create_net_from_spec
from sacred import Experiment

if opt.has_pycuda:
    from brainstorm.handlers import PyCudaHandler
    HANDLER = PyCudaHandler()
else:
    from brainstorm.handlers import default_handler
    HANDLER = default_handler

ex = Experiment('binding_dae')


@ex.config
def cfg():
    dataset = {
        'name': 'corners',
        'salt_n_pepper': 0.5,
        'train_set': 'train_single'  # train_multi or train_single
    }
    training = {
        'learning_rate': 0.01,
        'patience': 10,
        'max_epochs': 500
    }
    em = {
        'nr_iters': 10,
        'k': 3,
        'nr_samples': 1000,
        'e_step': 'expectation',  # expectation, expectation_pi, max, or max_pi
        'init_type': 'gaussian'   # gaussian, uniform, or spatial
    }
    network_spec = "F64"
    net_filename = 'Networks/binding_dae_{}_{}.h5'.format(
        dataset['name'],
        np.random.randint(0, 1000000))
    verbose = True


@ex.named_config
def random_search():
    network_spec = "F{act_func}{size}".format(
        act_func=str(np.random.choice(['r', 't', 's'])),
        size=np.random.choice([100, 250, 500, 1000]))
    training = {
        'learning_rate': float(10**np.random.uniform(-3, 0))}
    dataset = {
        'salt_n_pepper': float(np.random.randint(0, 10) / 10)}


@ex.capture(prefix='dataset')
def open_dataset(name):
    data_dir = os.environ.get('BRAINSTORM_DATA_DIR', './Datasets')
    filename = os.path.join(data_dir, name + '.h5')
    return h5py.File(filename, 'r')


@ex.capture(prefix='dataset')
def get_input_shape(train_set):
    with open_dataset() as f:
        return f[train_set]['default'].shape[2:]


@ex.capture
def create_network(network_spec, dataset):
    print("Network Specifications:", network_spec)
    with open_dataset() as f:
        in_shape = f[dataset['train_set']]['default'].shape[2:]
    net = create_net_from_spec('multi-label', in_shape, in_shape, network_spec,
                               use_conv=('C' in network_spec))
    return net


@ex.capture
def create_trainer(training, net_filename, verbose):
    trainer = bs.Trainer(bs.training.SgdStepper(training['learning_rate']),
                         verbose=verbose)
    trainer.train_scorers = [bs.scorers.Hamming()]
    trainer.add_hook(bs.hooks.StopOnNan())
    trainer.add_hook(bs.hooks.StopAfterEpoch(training['max_epochs']))
    trainer.add_hook(bs.hooks.MonitorScores('val_iter', trainer.train_scorers,
                                            name='validation'))
    trainer.add_hook(bs.hooks.EarlyStopper('validation.total_loss',
                                           patience=training['patience']))
    trainer.add_hook(bs.hooks.SaveBestNetwork('validation.total_loss',
                                              net_filename, criterion='min'))
    trainer.add_hook(bs.hooks.InfoUpdater(ex))
    if verbose:
        trainer.add_hook(bs.hooks.StopOnSigQuit())
        trainer.add_hook(bs.hooks.ProgressBar())
    return trainer


@ex.capture(prefix='dataset')
def get_data_iters(name, salt_n_pepper, train_set):
    with open_dataset(name) as f:
        train_size = int(0.9 * f[train_set]['default'].shape[1])
        train_data = f[train_set]['default'][:, :train_size]
        val_data = f[train_set]['default'][:, train_size:]

    train_iter = bs.data_iterators.AddSaltNPepper(
        bs.data_iterators.Minibatches(default=train_data, targets=train_data,
                                      batch_size=100),
        {'default': salt_n_pepper})

    val_iter = bs.data_iterators.AddSaltNPepper(
        bs.data_iterators.Minibatches(default=val_data, targets=val_data,
                                      batch_size=100),
        {'default': salt_n_pepper})
    return train_iter, val_iter


def get_test_data():
    with open_dataset() as f:
        test_groups = f['test']['groups'][:]
        test_data = f['test']['default'][:]
    return test_data, test_groups


def evaluate_groups(true_groups, predicted):
    idxs = np.where(true_groups != 0.0)
    score = adjusted_mutual_info_score(true_groups[idxs],
                                       predicted.argmax(1)[idxs])
    confidence = np.mean(predicted.max(1)[idxs])
    return score, confidence


@ex.capture
def load_best_net(net_filename):
    net = bs.Network.from_hdf5(net_filename)
    net.output_name = "Output.outputs.predictions"
    return net


@ex.capture(prefix='em')
def get_initial_groups(k, dims, init_type, _rnd, low=.25, high=.75):
    shape = (1, 1, dims[0], dims[1], 1, k)  # (T, B, H, W, C, K)
    if init_type == 'spatial':
        assert k == 3
        group_channels = np.zeros((dims[0], dims[1], 3))
        group_channels[:, :, 0] = np.linspace(0, 0.5, dims[0])[:, None]
        group_channels[:, :, 1] = np.linspace(0, 0.5, dims[1])[None, :]
        group_channels[:, :, 2] = 1.0 - group_channels.sum(2)
        group_channels = group_channels.reshape(shape)
    elif init_type == 'gaussian':
        group_channels = np.abs(_rnd.randn(*shape))
        group_channels /= group_channels.sum(5)[..., None]
    elif init_type == 'uniform':
        group_channels = _rnd.uniform(low, high, size=shape)
        group_channels /= group_channels.sum(5)[..., None]
    else:
        raise ValueError('Unknown init_type "{}"'.format(init_type))
    return group_channels


def get_likelihood(Y, T, group_channels):
    log_loss = T * np.log(Y.clip(1e-6, 1 - 1e-6)) + \
               (1 - T) * np.log((1 - Y).clip(1e-6, 1 - 1e-6))
    return np.sum(log_loss * group_channels)


@ex.capture(prefix='em')
def perform_e_step(T, Y, mixing_factors, e_step, k):
    loss = (T * Y + (1 - T) * (1 - Y)) * mixing_factors
    if e_step == 'expectation':
        group_channels = loss / loss.sum(5)[..., None]
    elif e_step == 'expectation_pi':
        group_channels = loss / loss.sum(5)[..., None]
        mixing_factors = group_channels.reshape(-1, k).sum(0)
        mixing_factors /= mixing_factors.sum()
    elif e_step == 'max':
        group_channels = (loss == loss.max(5)[..., None]).astype(np.float)
    elif e_step == 'max_pi':
        group_channels = (loss == loss.max(5)[..., None]).astype(np.float)
        mixing_factors = group_channels.reshape(-1, k).sum(0)
        mixing_factors /= mixing_factors.sum()
    else:
        raise ValueError('Unknown e_type: "{}"'.format(e_step))

    return group_channels, mixing_factors


@ex.command(prefix='em')
def reconstruction_clustering(network, input_data, true_groups, k, nr_iters):
    T, N, H, W, C = input_data.shape
    input_data = input_data[..., None]  # add a cluster dimension

    mixing_factors = np.ones((1, 1, 1, 1, k)) / k
    gamma = get_initial_groups(dims=(H, W))
    output_prior = np.ones_like(input_data) * 0.5

    gammas = np.zeros((nr_iters + 1, 1, H, W, C, k))
    likelihoods = np.zeros(2 * nr_iters + 1)
    scores = np.zeros((nr_iters + 1, 2))

    gammas[0:1] = gamma
    likelihoods[0] = get_likelihood(output_prior, input_data, gamma)
    scores[0] = evaluate_groups(true_groups.flatten(),
                                    gamma.reshape(-1, k))

    for j in range(nr_iters):
        X = gamma * input_data
        Y = np.zeros_like(X)

        # run the k copies of the autoencoder
        for _k in range(k):
            network.provide_external_data({'default': X[..., _k],
                                           'targets': input_data[..., 0]})
            network.forward_pass()
            Y[..., _k] = network.get(network.output_name).reshape((1, 1, H, W, C))

        # save the log-likelihood after the M-step
        likelihoods[2*j+1] = get_likelihood(Y, input_data, gamma)
        # perform an E-step
        gamma, mixing_factors = perform_e_step(input_data, Y, mixing_factors)
        # save the log-likelihood after the E-step
        likelihoods[2*j+2] = get_likelihood(Y, input_data, gamma)
        # save the resulting group-assignments
        gammas[j+1] = gamma[0]
        # save the score and confidence
        scores[j+1] = evaluate_groups(true_groups.flatten(),
                                          gamma.reshape(-1, k))
    return gammas, likelihoods, scores


@ex.command(prefix='em')
def evaluate(nr_samples, dump_results=None):
    network = load_best_net()
    test_data, test_groups = get_test_data()
    all_scores = []
    all_likelihoods = []
    all_gammas = []
    nr_samples = min(nr_samples, test_data.shape[1])
    for i in range(nr_samples):
        gammas, likelihoods, scores = reconstruction_clustering(
                network, test_data[:, i:i+1], test_groups[:, i:i+1])
        all_gammas.append(gammas)
        all_likelihoods.append(likelihoods)
        all_scores.append(scores)

    all_gammas = np.array(all_gammas)
    all_likelihoods = np.array(all_likelihoods)
    all_scores = np.array(all_scores)

    print('Average Score: {:.4f}'.format(all_scores[:, -1, 0].mean()))
    print('Average Confidence: {:.4f}'.format(all_scores[:, -1, 1].mean()))

    if dump_results is not None:
        import pickle
        with open(dump_results, 'wb') as f:
            pickle.dump((all_scores, all_likelihoods, all_gammas), f)
        print('wrote the results to {}'.format(dump_results))
    return all_scores[:, -1, 0].mean()


@ex.command
def draw_net(filename='net.png'):
    network = create_network()
    from brainstorm.tools import draw_network
    draw_network(network, filename)


@ex.automain
def run(net_filename):
    network = create_network()
    network.set_handler(HANDLER)
    trainer = create_trainer()
    train_iter, val_iter = get_data_iters()

    trainer.train(network, train_iter, val_iter=val_iter)

    ex.add_artifact(net_filename)

    ex.info['best_val_loss'] = float(np.min(trainer.logs['validation']['total_loss']))
    return evaluate()
