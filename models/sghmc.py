import argparse
import json
import time
import os
import logging

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def get_config(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dim',
        help='dimension of the 1d input vector',
        type=int,
        default=784,
        dest='input_dim',
    )
    parser.add_argument(
        '--split-val',
        help='number of samples for the validation set',
        type=int,
        default=10000,
        dest='splt_val',
    )
    parser.add_argument(
        '--train-batch-size',
        help='training minibatch size',
        type=int,
        default=500,
        dest='train_batch_size',
    )
    parser.add_argument(
        '--train-num-epochs',
        help='training minibatch size',
        type=int,
        default=800,
        dest='train_num_epochs',
    )
    parser.add_argument(
        '--logging-level',
        type=str,
        default='info',
        dest='logging_level',
    )
    parser.add_argument(
        '--logging-to-console',
        help='Print logs to console',
        action='store_true',
        default=True,
        dest='logging_to_console',
    )
    t = time.localtime()
    timestamp = (
        f'{t.tm_mon:02}{t.tm_mday:02}-'
        f'{t.tm_hour:02}-{t.tm_min:02}-{t.tm_sec:02}'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=timestamp,
        dest='model_name',
    )

    config = parser.parser_args(args)

    return config


def set_logging(config):
    logging_kwargs = {}
    if (config.logging_level == 'debug'):
        logging_kwargs['level'] = logging.DEBUG
    elif (config.logging_level == 'info'):
        logging_kwargs['level'] = logging.INFO
    else:
        logging_kwargs['level'] = logging.NOTSET

    if not config.logging_to_console:
        logging_kwargs['filename'] = f'{config.model_name}/log.txt'
        logging_kwargs['filemode'] = 'w'

    logging.basicConfig(**logging_kwargs)


def get_mnist_tf_dataset(config):
    split_val = config.split_val
    train_batch_size = config.train_batch_size
    train_num_epochs = config.train_num_epochs

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / np.iinfo(x_train.dtype).max
    x_test = x_test / np.iinfo(x_train.dtype).max

    x_train, x_val = np.split(x_train, [-split_val])
    y_train, y_val = np.split(y_train, [-split_val])

    dataset = {}

    dataset['training'] = (
        tf.data.Dataset.from_tensor_slices(x_train, y_train)
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .repeat(train_num_epochs).batch(train_batch_size)
    )

    dataset['validation'] = (
        tf.data.Dataset.from_tensor_slices(x_val, y_val)
        .repeat().batch(len(x_val))
    )

    dataset['test'] = (
        tf.data.Dataset.from_tensor_slices(x_test, y_test)
        .repeat().batch(len(x_test))
    )

    iterator = {}
    for split, ds in dataset.items():
        iterator[split] = ds.make_one_shot_iterator()

    return (dataset, iterator)


def get_sghmc_state_prior(var, lambda_var):
    return tfp.distributions.Normal(loc=0., scale=lambda_var) 


class BayesianNeuralNetwork:
    def __init__(
        self,
        config,
    ):
        os.mkdir(config.model_name)
        print('saving files @ {}.'.format(config.model_name))
        with open(f'{config.model_name}/config.json', 'w') as fp:
            json.dump(vars(config), fp)

        set_logging(config)
        logger = logging.getLogger(config.model_name)

        for option, value in vars(config).items():
            logger.info(f'{option}: {value}')

        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self._build_bnn()
            with tf.variable_scope('summary'):
                self._build_summary_ops()

    def run(self)
        with self.tf_graph.as_default():
            self.tf_session = tf.Session(config=self.tf_config)
            self.tf_session.run(tf.global_variable_initializer())

    def _build_bnn(self):
        self.var_A = tf.get_variable('A', shape=[10, 100])
        self.var_a = tf.get_variable('a', shape=[10])
        self.var_B = tf.get_variable('B', shape=[100, config.input_dim])
        self.var_b = tf.get_variable('b', shape=[100])
        self.sghmc_states = [
            self.var_A,
            self.var_a,
            self.var_B,
            self.var_b,
        ]
        self.var_lambda_A = tf.get_variable('lambda_A', shape=[1])
        self.var_lambda_a = tf.get_variable('lambda_a', shape=[1])
        self.var_lambda_B = tf.get_variable('lambda_B', shape=[1])
        self.var_lambda_b = tf.get_variable('lambda_b', shape=[1])
        self.gibbs_states = [
            self.var_lambda_A,
            self.var_lambda_a,
            self.var_lambda_B,
            self.var_lambda_b,
        ]
        # NOTE: tfp Gamma distribution
        # pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
        # Z = Gamma(alpha) beta**(-alpha),
        # concentration = alpha, alpha > 0,
        # rate = beta, beta > 0.

        alpha = tf.constant(1.0)
        beta = tf.constant(1.0)

        lambda_prior = tfp.distributions.Gamma(
            concentration=alpha, rate=beta,
        )

        A, a, B, b = self.sghmc_states
        lambda_A, lambda_a, var_lambda_B, var_lambda_b = self.gibbs_states

        P_A = 

        x = tf.placeholder(
            tf.float32,
            shape=[config.train_batch_size, config.input_dim]
        )

        layer_1 = tf.math.sigmoid(tf.linalg.matmul(B, x) + b)
        P_y = tf.math.exp(tf.linalg.matmul(A, layer_1) + a)
        unnormalized_log_prob = 


