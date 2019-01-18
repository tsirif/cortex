'''Generative adversarial networks with various objectives and penalties.

'''

import os
import functools
import math

from cortex._lib.config import CONFIG
from cortex._lib.exp import DEVICE
from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import register_plugin, ModelPlugin
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F

from .utils import (log_sum_exp, ms_ssim, update_decoder_args,
                    update_encoder_args, update_average_model)


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def get_boundary(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'H2', 'DV'):
        b = samples ** 2
    elif measure == 'X2':
        b = (samples / 2.) ** 2
    elif measure == 'W1':
        b = None
    else:
        raise_measure_error(measure)

    return b.mean()


def get_weight(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'DV', 'H2'):
        return samples ** 2
    elif measure == 'X2':
        return (samples / 2.) ** 2
    elif measure == 'W1':
        return None
    else:
        raise_measure_error(measure)


def generator_loss(q_samples, measure, loss_type=None):
    if not loss_type or loss_type == 'minimax':
        return get_negative_expectation(q_samples, measure)
    elif loss_type == 'non-saturating':
        return -get_positive_expectation(q_samples, measure)
    elif loss_type == 'boundary-seek':
        return get_boundary(q_samples, measure)
    else:
        raise NotImplementedError(
            'Generator loss type `{}` not supported. '
            'Supported: [None, non-saturating, boundary-seek]')


class GradientPenalty(ModelPlugin):

    def routine(self, inputs, penalty_type: str='contractive',
                penalty_amount: float=0.5):
        """

        Args:
            penalty_type: Gradient penalty type for the discriminator.
                {contractive}
            penalty_amount: Amount of gradient penalty for the discriminator.

        """
        if penalty_type == 'contractive':
            penalty = self.contractive_penalty(
                self.nets.network, inputs, penalty_amount=penalty_amount)
        else:
            raise NotImplementedError(penalty_type)

        if penalty:
            self.losses.network = penalty

    @staticmethod
    def _get_gradient(inp, output):
        gradient = autograd.grad(outputs=output, inputs=inp,
                                 grad_outputs=torch.ones_like(output),
                                 create_graph=True, retain_graph=True,
                                 only_inputs=True, allow_unused=True)[0]
        return gradient

    def contractive_penalty(self, network, input, penalty_amount=0.5):

        if penalty_amount == 0.:
            return

        if not isinstance(input, (list, tuple)):
            input = [input]

        input = [inp.detach() for inp in input]
        input = [inp.requires_grad_() for inp in input]

        with torch.set_grad_enabled(True):
            output = network(*input)
        gradient = self._get_gradient(input, output)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = (gradient ** 2).sum(1).mean()

        return penalty_amount * penalty

    def interpolate_penalty(self, network, input, penalty_amount=0.5):

        input = input.detach()
        input = input.requires_grad_()

        if len(input) != 2:
            raise ValueError('tuple of 2 inputs required to interpolate')
        inp1, inp2 = input

        try:
            epsilon = network.inputs.e.view(-1, 1, 1, 1)
        except AttributeError:
            raise ValueError('You must initiate a uniform random variable'
                             '`e` to use interpolation')
        mid_in = ((1. - epsilon) * inp1 + epsilon * inp2)
        mid_in.requires_grad_()

        with torch.set_grad_enabled(True):
            mid_out = network(mid_in)
        gradient = self._get_gradient(mid_in, mid_out)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = ((gradient.norm(2, dim=1) - 1.) ** 2).mean()

        return penalty_amount * penalty


class Discriminator(ModelPlugin):

    def build(self, discriminator_type: str='convnet', discriminator_args={}):
        """

        Args:
            discriminator_type: Discriminator network type.
            discriminator_args: Discriminator network arguments.

        """
        x_shape = self.get_dims('x', 'y', 'c')
        Encoder, discriminator_args = update_encoder_args(
            x_shape, model_type=discriminator_type,
            encoder_args=discriminator_args)
        discriminator = Encoder(x_shape, dim_out=1, **discriminator_args)
        self.nets.discriminator = discriminator

    def routine(self, real, fake, measure: str='GAN'):
        """

        Args:
            measure: GAN measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2 (squared
                Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        """
        X_P = real
        X_Q = fake
        E_pos, E_neg, P_samples, Q_samples = self.score(X_P, X_Q, measure)

        difference = E_pos - E_neg
        self.results.update(Scores=dict(Ep=P_samples.mean().item(),
                                        Eq=Q_samples.mean().item()))
        self.results['{} distance'.format(measure)] = difference.item()
        self.losses.discriminator = -difference

    def score(self, X_P, X_Q, measure):
        discriminator = self.nets.discriminator
        P_samples = discriminator(X_P)
        Q_samples = discriminator(X_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)

        return E_pos, E_neg, P_samples, Q_samples

    def visualize(self, real, fake, measure=None):
        X_P = real
        X_Q = fake

        E_pos, E_neg, P_samples, Q_samples = self.score(X_P, X_Q, measure)

        self.add_histogram(dict(fake=Q_samples.view(-1).data,
                                real=P_samples.view(-1).data),
                           name='discriminator output')


class SimpleDiscriminator(Discriminator):
    """
    Discriminator for 1d vectors.

    """

    def build(self, dim_in: int=None, discriminator_args=dict(dim_h=[200, 200])):
        """

        Args:
            dim_in (int): Input size
            dim_out (int): Output size
            classifier_args: Extra arguments for building the classifier

        """
        discriminator = FullyConnectedNet(dim_in, dim_out=1,
                                          **discriminator_args)
        self.nets.discriminator = discriminator


class Generator(ModelPlugin):

    def build(self, dim_z=64, generator_type: str='convnet',
              generator_args=dict(output_nonlinearity='tanh'),
              average_weight_model: float=0):
        """

        Args:
            generator_noise_type: Type of input noise for the generator.
            dim_z: Input noise dimension for generator.
            generator_type: Generator network type.
            generator_args: Generator network arguments.
            average_weight_model: If >0, keep a moving average model for testing.

        """
        x_shape = self.get_dims('x', 'y', 'c')

        Decoder, generator_args = update_decoder_args(
            x_shape, model_type=generator_type, decoder_args=generator_args)
        generator = Decoder(x_shape, dim_in=dim_z, **generator_args)
        self.nets.generator = generator

        awm = average_weight_model
        assert(awm >= 0 and 1 > awm)
        if awm > 0:
            self.nets.test_generator = generator.clone()
        else:
            self.nets.test_generator = generator

    def train_step(self, average_weight_model: float=0):
        super().train_step()
        if average_weight_model > 0:
            update_average_model(self.nets.test_generator, self.nets.generator,
                                 beta=average_weight_model)

    def routine(self, Z, measure: str=None, loss_type: str='non-saturating'):
        """

        Args:
            loss_type: Generator loss type.
                {non-saturating, minimax, boundary-seek}

        """
        discriminator = self.nets.discriminator

        X_Q = self.generate(Z)
        samples = discriminator(X_Q)

        g_loss = generator_loss(samples, measure, loss_type=loss_type)
        self.losses.generator = g_loss

    def generate(self, Z):
        _generated = self.generated
        if self._train:
            generator = self.nets.generator
        else:
            generator = self.nets.test_generator
        return generator(Z) if _generated is None else _generated

    @property
    def generated(self):
        ret = self._use_generated
        self._use_generated = None
        return ret

    @generated.setter
    def generated(self, generated_):
        self._use_generated = generated_

    def visualize(self, Z):
        X_Q = self.generate(Z)
        self.add_image(X_Q, name='generated')


class GeneratorEvaluator(ModelPlugin):
    """A module for testing a generator model."""

    @staticmethod
    def _default_inception_tar_filename():
        default_filename = 'frozen_inception_v1.tar.gz'
        _local_path = CONFIG.data_paths.get('local')
        if _local_path is not None:
            return os.path.join(os.abspath(_local_path), default_filename)
        return os.path.join('/tmp/cortex', default_filename)

    _default_filename = _default_inception_tar_filename()
    _num_of_kid_blocks = 8

    @staticmethod
    def pytorch_to_tf_device(device):
        try:
            device_id, device_num = str(device).split(':')
        except ValueError:
            device_id == 'cpu'
        if device_id == 'cpu':
            tf_device_str = "/cpu:0"
        elif device_id == 'cuda':
            tf_device_str = "/device:GPU:" + str(device_num)
        else:
            raise ValueError("Unknown pytorch to tensorflow device conversion.")
        return tf_device_str

    @classmethod
    def _get_inception_graph(cls, tar_filename):
        """Fetch inception graph tarball in a persistent `tarball_location`."""
        import tensorflow.contrib.gan.eval as tfgan
        return tfgan.get_graph_def_from_url_tarball(
            tfgan.INCEPTION_URL, tfgan.INCEPTION_FROZEN_GRAPH,
            os.path.abspath(tar_filename))

    def __init__(self):
        super().__init__()
        self.tf = None
        self.array_ops = None
        self.functional_ops = None
        self.tfgan = None

    def routine(self, reals, fakes,
                use_inception_score=True,
                use_fid=False, use_kid=False, use_ms_ssim=False,
                inception_tar_filename=_default_filename):
        """
            Args:
                use_inception_score:
                use_fid:
                use_kid:
                use_ms_ssim:
                inception_tar_filename:

        """
        # TODO Verify that this plugin works correctly!!!
        # NOTE: `reals` should not have any meaningful sorting in order to
        #       calculate a correct estimation of Kernel Inception Distance.
        reals_len = reals.size()[0]
        fakes_len = fakes.size()[0]

        if any((use_inception_score, use_fid, use_kid)):
            import tensorflow as tf
            self.tf = tf
            from tensorflow.python.ops import array_ops
            self.array_ops = array_ops
            from tensorflow.python.ops import functional_ops
            self.functional_ops = functional_ops
            tfgan = tf.contrib.gan.eval
            self.tfgan = tfgan

            with tf.device(GeneratorEvaluator.pytorch_to_tf_device(DEVICE)):
                inception_graph = self._get_inception_graph(inception_tar_filename)

                output_tensor = []
                output_tensor += [tfgan.INCEPTION_FINAL_POOL] if use_fid or use_kid else []
                output_tensor += [tfgan.INCEPTION_OUTPUT] if use_inception_score else []

                acts = self.precalc_inception_activations(reals if use_fid or use_kid else None,
                                                          fakes,
                                                          output_tensor, inception_graph)
                real_a, real_l, gen_a, gen_l = acts

                eval_scores = {}
                if use_inception_score:
                    is_fn = tfgan.classifier_score_from_logits
                    score = is_fn(gen_l)
                    eval_scores['IS'] = score

                if use_fid:
                    # Use O(n) estimation of FID (diagonal only calc of covariances)
                    # TODO perhaps use a hyperparameter to select available variant
                    fid_fn = tfgan.diagonal_only_frechet_classifier_distance_from_activations
                    fid = fid_fn(real_a, gen_a)
                    eval_scores['FID'] = fid

                if use_kid:
                    # Split data into 8 blocks and calculate estimations of
                    # KID and its std
                    # TODO perhaps introduce a hparam to control splits
                    kid_fn = tfgan.kernel_classifier_distance_and_std_from_activations
                    kid_fn = functools.partial(kid_fn,
                                               max_block_size=fakes_len // self._num_of_kid_blocks)
                    kid_mean, kid_std = kid_fn(real_a, gen_a)
                    eval_scores['KID'] = kid_mean
                    eval_scores['KID_std'] = kid_std

            # Execute collected scores
            sess = tf.Session()
            with sess.as_default():
                eval_scores = sess.run(eval_scores)

            # Log scores
            for key, value in eval_scores.items():
                self.results[key] = float(value)

        if use_ms_ssim:
            self.calc_ms_ssim(reals.cuda(), fakes.cuda())

    def precalc_inception_activations(self, reals, fakes, output_tensor,
                                      inception_net):
        dtype = tuple(self.tf.float32 for _ in output_tensor)

        classifier_fn = functools.partial(self.tfgan.run_inception,
                                          graph_def=inception_net,
                                          input_tensor=self.tfgan.INCEPTION_INPUT,
                                          output_tensor=output_tensor)

        def call_classifier(elems):
            return self.functional_ops.map_fn(fn=classifier_fn,
                                              elems=elems,
                                              dtype=dtype,
                                              parallel_iterations=1,
                                              back_prop=False,
                                              swap_memory=True,
                                              name='RunClassifier')

        def compute_activations(images):
            assert(type(images) == np.ndarray)
            assert(len(images.shape) == 4)
            assert(images.shape[1] == 3)
            assert(np.min(images[0]) >= -1 and np.max(images[0]) <= 1)
            images = self.tf.transpose(images, [0, 2, 3, 1])
            size = self.tfgan.INCEPTION_DEFAULT_IMAGE_SIZE
            images = self.tf.image.resize_bilinear(images, [size, size])
            #  images = tfgan._validate_images(images, size)
            num_splits = images.shape[0] // self.data.batch_size['test']
            images_list = self.array_ops.split(images,
                                               num_or_size_splits=num_splits)
            imgs = self.array_ops.stack(images_list)
            act = call_classifier(imgs)
            if len(output_tensor) == 2:
                final = self.array_ops.concat(self.array_ops.unstack(act[0]), 0)
                logits = self.array_ops.concat(self.array_ops.unstack(act[1]), 0)
            elif output_tensor[0] == self.tfgan.INCEPTION_FINAL_POOL:
                final = self.array_ops.concat(self.array_ops.unstack(act[0]), 0)
                logits = None
            else:
                final = None
                logits = self.array_ops.concat(self.array_ops.unstack(act[0]), 0)
            return final, logits

        real_a = None
        real_l = None
        gen_a, gen_l = compute_activations(fakes.numpy())
        if reals is not None:
            real_a, real_l = compute_activations(reals.numpy())

        return real_a, real_l, gen_a, gen_l

    def calc_ms_ssim(self, reals, fakes):
        # TODO Implement!!
        raise NotImplementedError("Use perhaps existing cortex implementation?")


class GAN(ModelPlugin):
    """
        Generative adversarial network.
        A generative adversarial network on images.
    """

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images')),
        train=dict(save_on_lowest='losses.gan')
    )

    def __init__(self):
        super().__init__()

        self.discriminator = Discriminator()
        penalty_contract = dict(nets=dict(network='discriminator'))
        self.penalty = GradientPenalty(contract=penalty_contract)
        self.generator = Generator()
        self.generator_eval = GeneratorEvaluator()

    def build(self, noise_type='normal', dim_z=64):
        """

        Args:
            noise_type: Distribution of input noise for generator.

        """
        self.add_noise('Z', dist=noise_type, size=dim_z)
        self.add_noise('E', dist='uniform', size=1)

        self.generator.build()
        self.discriminator.build()

    def train_step(self, generator_updates=1, discriminator_updates=1):
        """

        Args:
            generator_updates: Number of generator updates per step.
            discriminator_updates: Number of discriminator updates per step.

        """
        for _ in range(discriminator_updates):
            self.data.next()
            inputs, Z = self.inputs('inputs', 'Z')
            generated = self.generator.generate(Z)
            self.discriminator.routine(inputs, generated.detach())
            # TODO Hyperparameter to select where and how to combine gp
            # TODO {'real_penalty': 0.5, 'fake_penalty': 0, 'roth_penalty': 0}
            # TODO Write a method for that, instantiate as many GradientPenalty
            # models as needed, and alias properly its routine inputs (use
            # contract).
            self.penalty.routine(auto_input=True)
            self.optimizer_step()
            self.losses.clear()

        for _ in range(generator_updates):
            self.generator.train_step()

    def eval_step(self, eval_batch_size: int=4096):
        """
        Args:
            eval_batch_size: Sample size to compute generator evaluation with.
        """
        rounds = eval_batch_size // self.data.batch_size['test']
        reals = []
        fakes = []

        for _ in range(rounds):
            self.data.next()
            inputs, Z = self.inputs('inputs', 'Z')
            generated = self.generator.generate(Z)
            self.generator.generated = generated
            self.discriminator.routine(inputs, generated)
            self.penalty.routine(auto_input=True)
            self.generator.routine(auto_input=True)
            reals.append(inputs.cpu())
            fakes.append(generated.cpu())

        reals = torch.cat(reals)
        fakes = torch.cat(fakes)
        self.generator_eval.routine(reals, fakes)

    def visualize(self, images, Z):
        self.add_image(images, name='ground truth')
        generated = self.generator.generate(Z)
        self.generator.generated = generated
        self.discriminator.visualize(images, generated)
        self.generator.visualize(Z)


register_plugin(GAN)
