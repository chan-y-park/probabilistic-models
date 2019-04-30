import argparse
import logging

import tensorflow as tf


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
    parser.add_argument(
        '--num-leapfrog-steps',
        type=int,
        default=3,
        dest='num_leapfrog_steps',
    )
    parser.add_argument(
        '--hmc-step-size',
        type=float,
        default=0.1,
        dest='hmc_step_size',
    )
    parser.add_argument(
        '--hmc-num-burnin-steps',
        type=int,
        default=1e2,
        dest='hmc_num_burnin_steps',
    )
    parser.add_argument(
        '--hmc-num-simulation-steps',
        type=int,
        default=1e3,
        dest='hmc_num_simulation_steps',
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


