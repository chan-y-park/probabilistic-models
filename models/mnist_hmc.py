import json
import time
import os

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


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
        layer_2 = tf.linalg.matmul(A, layer_1) + a
        unnormalized_log_prob = (
            layer_2
            - lambda_A * tf.math.reduce_sum(tf.math.square(A))
            - lambda_a * tf.math.reduce_sum(tf.math.square(a))
            - lambda_B * tf.math.reduce_sum(tf.math.square(B))
            - lambda_b * tf.math.reduce_sum(tf.math.square(b))
        )


