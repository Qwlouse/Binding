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
        'max_epochs': 200
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
    return net


@ex.capture(prefix='em')
def get_initial_groups(k, dim, init_type, _rnd, low=.25, high=.75):
    shape = (1, 1, dim, dim, 1, k)  # (T, B, H, W, C, K)
    if init_type == 'spatial':
        assert k == 3
        group_channels = np.zeros((dim, dim, 3))
        group_channels[:, :, 0] = np.linspace(0, 0.5, dim).reshape(dim, 1)
        group_channels[:, :, 1] = np.linspace(0, 0.5, dim).reshape(1, dim)
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


@ex.command(prefix='em')
def evaluate(nr_iters, k, nr_samples, dump_results=None, dump_scores=None, nr_restarts=1):
    network = load_best_net()
    test_data, test_groups = get_test_data()
    scores = []
    confidences = []
    nr_samples = min(nr_samples, test_data.shape[1])
    in_feature_shape = test_data.shape[2:]
    all_scores = np.zeros((nr_restarts, nr_iters+1, nr_samples, 4))
    for r in range(nr_restarts):
        results = np.zeros((nr_iters + 1, nr_samples) + test_data.shape[2:] + (k,))
        likelihoods = np.zeros((2*nr_iters + 1, nr_samples))
        Y_prior = np.ones(test_data[:, 0:1].shape + (k,)) * 0.5
        scores = []
        confidences = []
        for i in range(nr_samples):
            mixing_factors = np.ones((1, 1, 1, 1, k)) / k

            x = test_data[:, i:i+1]   # (T, 1, 28, 28, 1)
            g = test_groups[:, i:i+1]  # (T, 1, 28, 28, 1)
            T = x[..., None]

            if np.sum(g) == 0:  # if that sample has no objects: ignore it
                continue
            group_channels = get_initial_groups(dim=x.shape[2])
            results[0:1, i:i+1, :] = group_channels

            # save the initial log-likelihood
            likelihoods[0, i] = get_likelihood(Y_prior, T, group_channels)
            all_scores[r, 0, i, :2] = evaluate_groups(g.flatten(), group_channels.reshape(-1, k))
            all_scores[r, 0, i, 2] = likelihoods[0, i]
            for j in range(nr_iters):
                X = group_channels * x[..., None]  # (T, 1, 28, 28, 1, 3)
                Y = np.zeros_like(X)  # (T, 1, 28, 28, 1, 3)

                # run the k copies of the autoencoder
                for _k in range(k):
                    network.provide_external_data({'default': X[..., _k],
                                                   'targets': x})
                    network.forward_pass()
                    Y[..., _k] = network.get(network.output_name).reshape((1, 1) + in_feature_shape)

                # save the log-likelihood after the M-step
                likelihoods[2*j+1, i] = get_likelihood(Y, T, group_channels)

                # perform an E-step
                group_channels, mixing_factors = perform_e_step(T, Y, mixing_factors)

                # save the log-likelihood after the E-step
                likelihoods[2*j+2, i] = get_likelihood(Y, T, group_channels)

                # save the resulting group-assignments
                results[j+1, i] = group_channels[0, 0]

                all_scores[r, j+1, i, :2] = evaluate_groups(g.flatten(), group_channels.reshape(-1, k))
                all_scores[r, j+1, i, 2] = likelihoods[2*j+1, i]
                all_scores[r, j+1, i, 3] = likelihoods[2*j+2, i]

            # evaluate the groups
            score, confidence = evaluate_groups(g.flatten(), group_channels.reshape(-1, k))
            confidences.append(confidence)
            scores.append(score)
        scores = np.array(scores)
        confidences = np.array(confidences)
        print("{:.4f} {:.4f} {:.4f} / {:.4f}".format(
            scores.min(), scores.mean(), scores.max(), confidences.mean()))

    if dump_results is not None:
        import pickle
        with open(dump_results, 'wb') as f:
            pickle.dump((scores, results, test_data[:, :nr_samples], confidences, likelihoods), f)
        print('wrote the results to {}'.format(dump_results))
    if dump_scores is not None:
        import pickle
        with open(dump_scores, 'wb') as f:
            pickle.dump(all_scores, f)
    return scores.mean(), confidences.mean()


@ex.command(prefix='em')
def convert(nr_iters, k, dest, nr_samples=1e12, splits=('training', 'test')):
    network = load_best_net()
    ds = open_dataset()
    seg_ds = h5py.File(dest + '_segregated.h5', 'w')
    rec_ds = h5py.File(dest + '_reconstructed.h5', 'w')
    assert nr_iters > 0

    for usage in splits:
        data = ds[usage]['default'][:]
        groups = ds[usage]['groups'][:]
        segregated = np.zeros(data.shape + (k,))
        reconstructed = np.zeros(data.shape + (k,))

        nr_samples = min(nr_samples, data.shape[1])
        in_feature_shape = data.shape[2:]
        scores = []
        confidences = []
        for i in range(nr_samples):
            mixing_factors = np.ones((1, 1, 1, 1, k)) / k

            x = data[:, i:i+1]   # (T, 1, 28, 28, 1)
            g = groups[:, i:i+1]  # (T, 1, 28, 28, 1)
            T = x[..., None]

            if np.sum(g) == 0:  # if that sample has no objects: ignore it
                continue
            group_channels = get_initial_groups(dim=x.shape[2])

            for j in range(nr_iters):
                X = group_channels * x[..., None]  # (T, 1, 28, 28, 1, 3)
                Y = np.zeros_like(X)  # (T, 1, 28, 28, 1, 3)

                # run the k copies of the autoencoder
                for _k in range(k):
                    network.provide_external_data({'default': X[..., _k],
                                                   'targets': x})
                    network.forward_pass()
                    Y[..., _k] = network.get(network.output_name).reshape((1, 1) + in_feature_shape)

                # perform an E-step
                group_channels, mixing_factors = perform_e_step(T, Y, mixing_factors)

            segregated[:, i] = group_channels * x[..., None]
            reconstructed[:, i] = Y
            score, confidence = evaluate_groups(g.flatten(), group_channels.reshape(-1, k))
            confidences.append(confidence)
            scores.append(score)
        scores = np.array(scores)
        confidences = np.array(confidences)
        print("{:.4f} {:.4f} {:.4f} / {:.4f}".format(
            scores.min(), scores.mean(), scores.max(), confidences.mean()))

        seg_usg = seg_ds.create_group(usage)
        seg_usg.create_dataset('default', data=segregated, compression='gzip', chunks=(1, 100) + in_feature_shape + (1,))
        seg_usg.create_dataset('groups', data=groups, compression='gzip', chunks=(1, 100) + in_feature_shape)

        rec_usg = rec_ds.create_group(usage)
        rec_usg.create_dataset('default', data=reconstructed, compression='gzip', chunks=(1, 100) + in_feature_shape + (1,))
        rec_usg.create_dataset('groups', data=groups, compression='gzip', chunks=(1, 100) + in_feature_shape)

        if 'targets' in ds[usage]:
            rec_usg.create_dataset('targets', data=ds[usage]['targets'][:], compression='gzip', chunks=(1, 100, 1))
            seg_usg.create_dataset('targets', data=ds[usage]['targets'][:], compression='gzip', chunks=(1, 100, 1))

    ds.close()
    seg_ds.close()
    rec_ds.close()


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
    score, confidence = evaluate()
    ex.info['confidence'] = confidence
    return score
