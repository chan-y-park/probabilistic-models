import json
import time
import os

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def get_sghmc_state_prior(var, lambda_var):
    return tfp.distributions.Normal(loc=0., scale=lambda_var) 


class HyperParameterLambda:
    def __init__(
        self,
        theta,
        initial_lambda,
    ):
        self.theta = theta
        self.var_lambda = tf.get_variable(
            'lambda_' + theta.name,
            shape=[1],
            initializer=tf.constant(initial_lambda),
        )
        self.var_alpha = tf.get_variable(
            'lambda_' + theta.name + '_alpha',
            shape=[1],
            initializer=tf.initializers.ones,
        )
        self.var_beta = tf.get_variable(
            'lambda_' + theta.name + '_beta',
            shape=[1],
            initializer=tf.initializers.ones,
        )

    def sample(self):
        # First update the Gamma distribution parameters.
        # alpha <- alpha + n/2,
        # beta <- beta + 1/2 * \sum_{i=1}^{n} (x_i - \mu)^2.
        tf.assign_add(
            self.var_alpha,
            0.5 * tf.size(self.theta)
        )
        tf.assign_add(
            self.var_beta,
            0.5 * tf.math.reduce_sum(tf.math.square(self.theta))
        )
        # Then perform a Gibbs sampling.
        # NOTE: tfp Gamma distribution
        # pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
        # Z = Gamma(alpha) beta**(-alpha),
        # concentration = alpha, alpha > 0,
        # rate = beta, beta > 0.
        posterior = tfp.distributions.Gamma(
            concentration=self.var_alpha, rate=self.var_beta,
        )
        tf.assign(
            self.var_lambda,
            posterior.sample(),
        )


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

    def _build_bnn(self):
        # TODO
        (initial_lambda_A,
         initial_lambda_a,
         initial_lambda_B,
         initial_lambda_b) = tfp.distributions.Gamma(
            concentration=1.0, rate=1.0,
        ).sample(4)
        self.var_A = tf.get_variable(
            'A',
            shape=[10, 100],
            dtype=tf.float32,
            initializer=theta_initializer(initial_lambda_A),
        )
        self.var_a = tf.get_variable(
            'a',
            shape=[10],
            dtype=tf.float32,
            initializer=theta_initializer(initial_lambda_a),
        )
        self.var_B = tf.get_variable(
            'B',
            shape=[100, config.input_dim],
            dtype=tf.float32,
            initializer=theta_initializer(initial_lambda_B),
        )
        self.var_b = tf.get_variable(
            'b',
            shape=[100],
            dtype=tf.float32,
            initializer=theta_initializer(initial_lambda_b),
        )
        self.hmc_states = [
            self.var_A,
            self.var_a,
            self.var_B,
            self.var_b,
        ]
        self.var_lambda_A = HyperParameterLambda(self.var_A)
        self.var_lambda_a = HyperParameterLambda(self.var_a)
        self.var_lambda_B = HyperParameterLambda(self.var_B)
        self.var_lambda_b = HyperParameterLambda(self.var_b)
        self.gibbs_states = [
            self.var_lambda_A,
            self.var_lambda_a,
            self.var_lambda_B,
            self.var_lambda_b,
        ]

        A, a, B, b = self.hmc_states
        lambda_A, lambda_a, var_lambda_B, var_lambda_b = self.gibbs_states

        self.images = tf.placeholder(
            dtype=tf.float32,
            shape=[config.train_batch_size, config.input_dim],
        )
        
        self.labels = tf.placeholder(
            dtype=tf.int32,
            shape=[config.train_batch_size, 10],
        )

        layer_1 = tf.math.sigmoid(tf.linalg.matmul(B, x) + b)
        layer_2 = tf.linalg.matmul(A, layer_1) + a
        unnormalized_log_prob = (
            tf.linalg.matmul(self.labels, layer_2)
            - lambda_A * tf.math.reduce_sum(tf.math.square(A))
            - lambda_a * tf.math.reduce_sum(tf.math.square(a))
            - lambda_B * tf.math.reduce_sum(tf.math.square(B))
            - lambda_b * tf.math.reduce_sum(tf.math.square(b))
        )
        # NOTE: See tfp.python.mcmc._leapfrog_integrator_one_step
        # for setting mass parameters of HMC momentum update.
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=config.num_leapfrog_steps,
            step_size=config.hmc_step_size,
#            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
#                num_adaptation_steps=int(config.hmc_num_burnin_steps * 0.8),
#            ),
        )
        self.current_states = [
            tf.placeholder(
                dtype=tf.float32,
                shape=theta.shape,
            ) for theta in self.hmc_states
        ]
        self.next_states, self.kernel_results = tfp.mcmc.sample_chain(
            num_results=config.hcm_num_simulation_steps,
            num_burnin_steps=config.hmc_num_burnin_steps,
            current_state=self.sghmc_states,
            kernel=hmc,
        )

    def run(self, num_iterations)
        with self.tf_graph.as_default():
            self.tf_session = tf.Session(config=self.tf_config)
            self.tf_session.run(tf.global_variable_initializer())
            for t in range(num_iterations):
                # TODO: Read dataset
                feed_dict = {x: mnist_images}

                # HMC
                for i_theta, theta in enumerate(self.current_states):
                    feed_dict[theta] = self.hmc_states[i_theta]
                fetch_dict = {
                    'next_states': self.next_states,
                }
                rd = self.tf_session.run(
                    fetches=fetch_dict,
                    feed_dict=feed_dict,
                )
                for i_theta, theta in enumerate(self.current_states):
                    tf.assign(
                        theta,
                        rd['next_states'][i_theta],
                    )

                # Gibbs
                for var_lambda in self.gibbs_states:
                    var_lambda.sample()


