'''Generative adversarial networks with various objectives and penalties.

'''

import copy
import os
import functools
import logging
import math

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import register_plugin, ModelPlugin
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F

from .utils import (log_sum_exp, update_decoder_args,
                    update_encoder_args, update_average_model)


logger = logging.getLogger(__name__)


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
        gradient = gradient.view(gradient.size(0), -1)
        penalty = gradient.pow(2).sum(-1).mean()

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
        self.results.update(Scores=dict(Ep=P_samples.mean(),
                                        Eq=Q_samples.mean()))
        self.results['{} distance'.format(measure)] = difference
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
            self.nets.test_generator = copy.deepcopy(generator)
        else:
            self.nets.test_generator = generator

        # First `self.generate(Z)` necessarily will generate something
        self._use_generated = None

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
        if self.is_training:
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
    INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
    INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
    INCEPTION_INPUT = 'Mul:0'
    INCEPTION_OUTPUT = 'logits:0'
    INCEPTION_FINAL_POOL = 'pool_3:0'
    INCEPTION_DEFAULT_IMAGE_SIZE = 299

    @staticmethod
    def _default_inception_tar_filename():
        from cortex._lib.config import CONFIG
        default_filename = 'frozen_inception_v1.tar.gz'
        _local_path = CONFIG.data_paths.get('local')
        if _local_path is not None and os.path.isdir(_local_path):
            return os.path.join(os.path.abspath(_local_path), default_filename)
        return os.path.join(os.path.abspath(os.path.curdir), default_filename)

    @staticmethod
    def _get_tf_device():
        from cortex._lib.exp import DEVICE
        try:
            device_id, device_num = str(DEVICE).split(':')
        except ValueError:
            device_id = 'cpu'
        if device_id == 'cpu':
            tf_device_str = "/cpu:0"
        elif device_id == 'cuda':
            tf_device_str = "/device:GPU:" + str(device_num)
        else:
            raise ValueError("Unknown pytorch to tensorflow device conversion.")
        return tf_device_str

    def _get_inception_graph(self, tar_filename):
        """Fetch inception graph tarball in a persistent `tarball_location`."""
        return self.tfgan.get_graph_def_from_url_tarball(
            self.INCEPTION_URL, self.INCEPTION_FROZEN_GRAPH,
            os.path.abspath(tar_filename))

    def build(self, inception_path=None):
        """
            Args:
                inception_path: Contains the persistent path to frozen inception_v1.

        """
        logger.info("GeneratorEvaluator. importing tensorflow...")
        import tensorflow as tf
        self.tf = tf
        from tensorflow.python.ops import array_ops
        self.array_ops = array_ops
        from tensorflow.python.ops import functional_ops
        self.functional_ops = functional_ops
        from tensorflow.python.ops import math_ops
        self.math_ops = math_ops
        self.tfgan = tf.contrib.gan.eval

        self.tf_device_name = self._get_tf_device()
        inception_path = inception_path or self._default_inception_tar_filename()
        logger.info("GeneratorEvaluator. inception_v1 path: %s", inception_path)
        with self.tf.device(self.tf_device_name):
            self.inception_graph = self._get_inception_graph(inception_path)

    def routine(self, reals, fakes,
                use_inception_score=False,
                use_fid=False,
                use_kid=False, kid_splits=1,
                inception_batch_size=None,
                use_ms_ssim=False,
                mock_GE=False):
        """
            Args:
                use_inception_score: True, to calculate inception score (IS).
                use_fid: True, to calculcate Frechet inception distance (FID).
                use_kid: True, to calculate kernel inception distance (KID).
                kid_splits: Number of splits to block estimate MMD in KID.
                inception_batch_size: Batch size to calculate inception activations with.
                use_ms_ssim: True, to calculate multi-scale structure similarity.
                mock_GE: True, to return mock values (DEBUGGING).

        """
        # NOTE: `reals` should not have any meaningful sorting in order to
        #       calculate a correct estimation of Kernel Inception Distance.
        if not any((use_inception_score, use_fid, use_kid, use_ms_ssim)):
            return

        eval_scores = {}
        reals_len = reals.size(0)
        fakes_len = fakes.size(0)
        assert(reals_len == fakes_len)
        logger.debug("GeneratorEvaluator. Calculating using %d samples.", reals_len)

        if mock_GE:
            if use_inception_score:
                self.results.IS = 3.
            if use_fid:
                self.results.FID = 0.4
            if use_kid:
                self.results.KID = 0.4
                self.results.KID_std = 0.1
            if use_ms_ssim:
                self.results.MS_SSIM = 3
                self.results.MS_SSIM_std = 0.5
            return

        reals_np = reals.numpy()
        fakes_np = fakes.numpy()

        if any((use_inception_score, use_fid, use_kid)):
            output_tensor = []
            output_tensor += [self.INCEPTION_FINAL_POOL] if use_fid or use_kid else []
            output_tensor += [self.INCEPTION_OUTPUT] if use_inception_score else []

            acts = self.precalc_inception_activations(reals_np if use_fid or use_kid else None,
                                                        fakes_np,
                                                        output_tensor, self.inception_graph,
                                                        inception_batch_size or self.data.batch_size['test'])
            real_a, real_l, gen_a, gen_l = acts

            if use_inception_score:
                is_fn = self.tfgan.classifier_score_from_logits
                score = is_fn(gen_l)
                eval_scores['IS'] = score

            if use_fid:
                # TODO perhaps use a hyperparameter to select available variant
                fid_fn = self.tfgan.frechet_classifier_distance_from_activations
                fid = fid_fn(real_a, gen_a)
                eval_scores['FID'] = fid

            if use_kid:
                # Split data into 8 blocks and calculate estimations of KID and its std
                kid_fn = self.tfgan.kernel_classifier_distance_and_std_from_activations
                kid_fn = functools.partial(kid_fn,
                                           max_block_size=math.ceil(fakes_len / kid_splits))
                kid_mean, kid_std = kid_fn(real_a, gen_a)
                eval_scores['KID'] = kid_mean
                if kid_splits > 1:
                    # FIXME: This patches a bug in the calculation of KID's std by tfgan
                    # Remove when it is corrected
                    kid_var = kid_splits * self.math_ops.square(kid_std)
                    eval_scores['KID_std'] = self.math_ops.sqrt(kid_var)

        if use_ms_ssim:
            ms_ssims = self.calc_ms_ssim(fakes_np, reals_np,
                                            batch_size=self.data.batch_size['test'])
            assert(self.array_ops.rank(ms_ssims) == 1)  # XXX
            assert(self.array_ops.shape(ms_ssims)[0] == reals_len)
            mn = self.math_ops.reduce_mean(ms_ssims)
            var = self.math_ops.square(ms_ssims - mn) / (reals_len - 1)
            std = self.math_ops.sqrt(var)
            eval_scores['MS_SSIM'] = mn
            eval_scores['MS_SSIM_std'] = std

        # Execute collected scores
        sess = self.tf.Session()
        with sess.as_default():
            eval_scores = sess.run(eval_scores)

        # Log scores
        for key, value in eval_scores.items():
            self.results[key] = value.item()  # numpy's `item`

    def precalc_inception_activations(self, reals, fakes, output_tensor,
                                      inception_net, batch_size=32):
        dtype = list(self.tf.float32 for _ in output_tensor)

        classifier_fn = functools.partial(self.tfgan.run_inception,
                                          graph_def=inception_net,
                                          input_tensor=self.INCEPTION_INPUT,
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
            # Permute ranks so that "channel" is the last one
            images = self.tf.transpose(images, [0, 2, 3, 1])
            # Resize to inception's input size
            size = self.INCEPTION_DEFAULT_IMAGE_SIZE
            images = self.tf.image.resize_bilinear(images, [size, size])
            # Calculate number of batches for computation efficiency
            num_batches = images.shape[0] // batch_size
            assert(images.shape[0] == batch_size * num_batches)
            images_list = self.array_ops.split(images,
                                               num_or_size_splits=num_batches)
            imgs = self.array_ops.stack(images_list)
            # Pass them through inception
            act = call_classifier(imgs)
            # Post-process outputs according to which ones where required
            if len(output_tensor) == 2:
                final = self.array_ops.concat(self.array_ops.unstack(act[0]), 0)
                logits = self.array_ops.concat(self.array_ops.unstack(act[1]), 0)
            elif output_tensor[0] == self.INCEPTION_FINAL_POOL:
                final = self.array_ops.concat(self.array_ops.unstack(act[0]), 0)
                logits = None
            else:
                final = None
                logits = self.array_ops.concat(self.array_ops.unstack(act[0]), 0)
            return final, logits

        real_a = None
        real_l = None
        gen_a, gen_l = compute_activations(fakes)
        if reals is not None:
            real_a, real_l = compute_activations(reals)

        return real_a, real_l, gen_a, gen_l

    def calc_ms_ssim(self, fakes, reals, batch_size=32):
        assert(type(fakes) == np.ndarray)
        assert(len(fakes.shape) == 4)
        assert(fakes.shape[1] == 3)
        assert(np.min(fakes[0]) >= -1 and np.max(fakes[0]) <= 1)
        assert(type(reals) == np.ndarray)
        assert(len(reals.shape) == 4)
        assert(reals.shape[1] == 3)
        assert(np.min(reals[0]) >= -1 and np.max(reals[0]) <= 1)
        # Permute ranks so that "channel" is the last one
        fakes = self.tf.transpose(fakes, [0, 2, 3, 1])
        reals = self.tf.transpose(reals, [0, 2, 3, 1])
        # Calculate number of batches for computation efficiency
        num_batches = fakes.shape[0] // batch_size
        assert(fakes.shape[0] == batch_size * num_batches)
        fakes_list = self.array_ops.split(fakes,
                                          num_or_size_splits=num_batches)
        fake_imgs = self.array_ops.stack(fakes_list)
        reals_list = self.array_ops.split(reals,
                                          num_or_size_splits=num_batches)
        real_imgs = self.array_ops.stack(reals_list)

        ms_ssim_fn = self.tf.image.ssim_multiscale
        res = self.functional_ops.map_fn(fn=lambda x: ms_ssim_fn(x[0], x[1], max_val=2),
                                         elems=(fake_imgs, real_imgs),
                                         parallel_iterations=1,
                                         back_prop=False,
                                         swap_memory=True,
                                         name='RunMSSSIM')

        ms_ssims = self.array_ops.concat(self.array_ops.unstack(res), 0)
        return ms_ssims


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

        self.discriminator.build()
        self.generator.build()
        self.generator_eval.build()

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

    def eval_step(self, score_sample_size: int=128*39):
        """
        Args:
            score_sample_size: Sample size to compute generator evaluation with.

        """
        batch_size = self.data.batch_size['test']
        assert(self.data.batch_size['train'] == self.data.batch_size['test'])  # XXX
        rounds = score_sample_size // batch_size
        reals = []
        fakes = []

        try:
            for _i in range(rounds):
                self.data.next()
                inputs, Z = self.inputs('inputs', 'Z')
                generated = self.generator.generate(Z)
                self.discriminator.routine(inputs, generated)
                self.penalty.routine(auto_input=True)
                self.generator.generated = generated
                self.generator.routine(auto_input=True)
                reals.append(inputs.detach().cpu())
                fakes.append(generated.detach().cpu())

        finally:
            # Calculate generator evaluation scores on "same" sample sizes
            # If `StopIteration` has been raised before all rounds have been
            # completed don't use the accumulated "reals" and "fakes"
            if _i == rounds - 1:
                reals = torch.cat(reals)
                fakes = torch.cat(fakes)
                self.generator_eval.routine(reals, fakes)

    def visualize(self, images, Z):
        self.add_image(images, name='ground truth')
        generated = self.generator.generate(Z)
        self.discriminator.visualize(images, generated)
        self.generator.generated = generated
        self.generator.visualize(Z)


register_plugin(GAN)
