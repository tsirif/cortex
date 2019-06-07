'''Module for training.

'''

import logging
import pprint
import sys
import time
import warnings

from . import (exp, random, viz)
from .utils import (convert_to_numpy, summarize_results, update_dict_of_lists)
from .viz import plot

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.train')


def train_epoch(model, epoch, full_eval_during_train, validate_batches=0,
                autotune_on=False,
                data_mode='train', use_pbar=True, fastdebug=None):
    model.train_loop(epoch, data_mode=data_mode, use_pbar=use_pbar,
                     validate_batches=validate_batches, autotune_on=autotune_on,
                     fastdebug=fastdebug)

    if full_eval_during_train:
        return eval_epoch(model, epoch, data_mode=data_mode, use_pbar=use_pbar)

    results = summarize_results(model._all_epoch_results)
    return convert_to_numpy(results)


def eval_epoch(model, epoch, data_mode='train', use_pbar=True, fastdebug=None):
    model.eval_loop(epoch, data_mode=data_mode, use_pbar=use_pbar,
                    fastdebug=fastdebug)
    results = summarize_results(model._all_epoch_results)
    return convert_to_numpy(results)


def test_epoch(model, epoch, data_mode='test', use_pbar=True, test_seed=None,
               fastdebug=None):
    train_prg_state = None
    if test_seed is not None:
        train_prg_state = random.get_prgs_state()
        random.reseed(test_seed)

    results = eval_epoch(model, epoch, data_mode=data_mode, use_pbar=use_pbar,
                         fastdebug=fastdebug)

    if train_prg_state is not None:
        random.set_prgs_state(train_prg_state)

    return results


def visualize_epoch(model, data_mode='test', test_seed=None):
    train_prg_state = None
    if test_seed is not None:
        train_prg_state = random.get_prgs_state()
        random.reseed(test_seed)

    model.data.reset(make_pbar=False, mode=data_mode)
    model.data.next()
    model.viz.clear()
    model.visualize(auto_input=True)

    if train_prg_state is not None:
        random.set_prgs_state(train_prg_state)


def display_results(train_results, valid_results, test_results,
                    epoch, epochs, epoch_time, total_time,
                    sep='\n\t\t'):
    if epochs and epoch:
        print('\n\tEpoch {} / {} took {:.3f}s. Total time: {:.2f}s'
              .format(epoch, epochs, epoch_time, total_time))

    times = train_results.pop('times', None)
    if times:
        time_strs = ['{}: {:.3f}'.format(k, v[0]) for k, v in times.items()]
        print('\tAvg update times::{}'.format(sep) + sep.join(time_strs))

    train_losses = train_results.pop('losses')
    test_losses = test_results.pop('losses', None)

    loss_strs = ['{}: {:.4f} ({:.4f})'
                 .format(k, train_losses[k][0], train_losses[k][1])
                 for k in train_losses.keys()]
    print('\tAvg (std) train loss::{}'.format(sep) + sep.join(loss_strs))
    if test_losses:
        loss_strs = ['{}: {:.4f} ({:.4f})'
                     .format(k, test_losses[k][0], test_losses[k][1])
                     for k in test_losses.keys()]
        print('\tAvg (std) test loss::{}'.format(sep) + sep.join(loss_strs))
    else:
        print('\tAvg (std) test loss:: -')

    tunables = train_results.pop('tunables', None)
    print('\n\tAvg (std) train results')
    print(  '\t-----------------------')
    for k, v_train in train_results.items():
        if isinstance(v_train, dict):
            print('\t{}::{}'.format(k, sep) + sep.join(['{}: {:.4f} ({:.4f})'
                  .format(k_, v[0], v[1]) for k_, v in v_train.items()]))
        else:
            print('\t{}: {:.4f} ({:.4f})'.format(k, v_train[0], v_train[1]))
    if tunables:
        print('\ttunables::{}'.format(sep) + sep.join(['{}: {:.4f}'
              .format(k_, v[0]) for k_, v in tunables.items()]))

    if valid_results:
        print('\n\tAvg validation results')
        print(  '\t----------------------')
        for k, v_valid in valid_results.items():
            if isinstance(v_valid, dict):
                print('\t{}::{}'.format(k, sep) + sep.join(['{}: {:.4f}'
                      .format(k_, v[0]) for k_, v in v_valid.items()]))
            else:
                print('\t{}: {:.4f}'.format(k, v_valid[0]))

    if test_results:
        print('\n\tAvg (std) test results')
        print(  '\t----------------------')
        for k, v_test in test_results.items():
            if isinstance(v_test, dict):
                print('\t{}::{}'.format(k, sep) + sep.join(['{}: {:.4f} ({:.4f})'
                      .format(k_, v[0], v[1]) for k_, v in v_test.items()]))
            else:
                print('\t{}: {:.4f} ({:.4f})'.format(k, v_test[0], v_test[1]))


def align_summaries(epoch, summary):
    for k in summary.keys():
        v = summary[k]
        if isinstance(v, dict):
            for vk in v.keys():
                v[vk] = v[vk] + [v[vk][-1]] * (epoch - len(v[vk]))
        else:
            summary[k] = v + [v[-1]] * (epoch - len(v))


def save_best(epoch, model, train_results, best, save_on_best, save_on_lowest):
    flattened_results = {}
    for k, v in train_results.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                flattened_results[k + '.' + k_] = v_
        else:
            flattened_results[k] = v
    if save_on_best in flattened_results:
        # TODO(Devon) This needs to be fixed.
        # when train_for is set, result keys vary per epoch
        # if save_on_best not in flattened_results:
        #    raise ValueError('`save_on_best` key `{}` not found.
        #  Available: {}'.format(
        #        save_on_best, tuple(flattened_results.keys())))
        current = flattened_results[save_on_best]
        if not best:
            found_best = True
        elif save_on_lowest:
            found_best = current < best
        else:
            found_best = current > best
        if found_best:
            best = current
            exp.save(model, prefix='best_' + save_on_best)
            msg = 'Found best {0} on epoch {1}: {2[0]:.5f} ({2[1]:.4f})'.format(
                save_on_best, epoch, best)
            print(msg)

        return best

    warnings.warn("Key '{}' was not found in flattened train results.\n"
                  "Flattened train result keys:\n{}".format(
                      save_on_best, tuple(flattened_results.keys())),
                  RuntimeWarning)


def main_loop(model, fastdebug, epochs=500, validate_batches=0, autotune_on=False,
              archive_every=10, test_every=1, visualize_every=1,
              save_on_best=None, save_on_lowest=False, save_on_highest=False,
              full_eval_during_train=False,
              train_mode='train', test_mode='test', eval_only=False,
              pbar_off=False):
    '''

    Args:
        epochs: Number of epochs.
        validate_batches: How many batches to be used in each epoch to validate
            the model in the beginning of its training loop.
        autotune_on: If True, it will try to autotune hyperparameters.
        archive_every: Period of epochs for writing checkpoints.
        test_every: Period of epochs for performing tests.
        visualize_every: Period of epochs for visualizing model results.
        save_on_best: Name of the key in results to track
            for saving the best model.
        save_on_lowest: Saves when lowest of `save_on_best` result is found.
        save_on_highest: Saves when highest of `save_on_best` result is found.
        full_eval_during_train: If True, perform one full pass on training set
            to evaluate model during training.
        train_mode: Mode of data to be used during training.
        test_mode: Mode of data to be used during testing.
        eval_only: Test on data only (no training).
        pbar_off: Turn off the progressbar.

    '''
    info = pprint.pformat(exp.ARGS)

    logger.info('Starting main loop.')

    if viz.visualizer:
        viz.visualizer.text(info, env=exp.NAME, win='info')

    if eval_only:
        test_results = test_epoch(model, 'Testing',
                                  data_mode=test_mode, use_pbar=False,
                                  fastdebug=fastdebug)
        display_results(test_results, dict(), dict(),
                        'Evaluation', None, None, None)
        exit(0)

    best = None
    if not isinstance(epochs, int):
        epochs = epochs['epochs']
    epoch = exp.INFO['epoch']
    first_epoch = epoch
    total_time = 0.

    if fastdebug:
        logger.warning("Overwriting `epochs` in order to fast debug for %d loops.",
                       fastdebug)
        epochs = epoch + fastdebug

    while epoch < epochs:
        try:
            start_time = time.time()

            # EPOCH INCREMENT
            exp.INFO['epoch'] += 1
            epoch = exp.INFO['epoch']
            if pbar_off:
                print("Training epoch %d / %d", epoch, epochs)

            # TRAINING & VALIDATION
            train_results_ = train_epoch(
                model, epoch, full_eval_during_train,
                validate_batches=validate_batches, autotune_on=autotune_on,
                data_mode=train_mode, use_pbar=not(pbar_off),
                fastdebug=fastdebug)

            # TRAINING AND VALIDATION SUMMARY
            valid_results_ = train_results_.pop('validate', dict())
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)
            if 'validate' not in exp.SUMMARY:  # XXX
                exp.SUMMARY['validate'] = dict()
            update_dict_of_lists(exp.SUMMARY['validate'], **valid_results_)

            # TESTING
            is_testing_epoch = (test_every and (epoch - 1) % test_every == 0) or \
                epoch == epochs
            test_results_ = dict()
            if is_testing_epoch:
                if pbar_off:
                    print("Evaluating epoch %d / %d", epoch, epochs)
                test_results_ = test_epoch(model, epoch, data_mode=test_mode,
                                           use_pbar=not(pbar_off),
                                           test_seed=exp.INFO.get('test_seed'),
                                           fastdebug=fastdebug)
                update_dict_of_lists(exp.SUMMARY['test'], **test_results_)

            # VISUALIZING
            is_viz_epoch = (visualize_every and (epoch - 1) % visualize_every == 0) or \
                epoch == epochs
            if is_viz_epoch:
                visualize_epoch(model, test_seed=exp.INFO.get('test_seed'))

            # SAVE IF BEST MODEL FOUND
            if save_on_best or save_on_highest or save_on_lowest:
                best = save_best(epoch, model, test_results_,
                                 best, save_on_best, save_on_lowest)

            # FINISH TOTAL SUMMARY
            align_summaries(epoch, exp.SUMMARY['train'])
            align_summaries(epoch, exp.SUMMARY['validate'])
            align_summaries(epoch, exp.SUMMARY['test'])

            epoch_time = time.time() - start_time
            total_time += epoch_time

            # LIVE PLOT VISUALIZATION
            plot(epoch, init=(epoch == first_epoch + 1))
            if is_viz_epoch:
                model.viz.show(epoch)

            # SHELL DISPLAY SUMMARY
            display_results(train_results_, valid_results_, test_results_,
                            epoch, epochs,
                            epoch_time, total_time)

            # CHECKPOINT MODEL
            if archive_every:
                if (epoch - 1) % archive_every == 0:
                    exp.save(model, prefix=epoch)
            else:
                exp.save(model, prefix='last')

        except KeyboardInterrupt:
            def stop_training_query():
                while True:
                    try:
                        response = input('Keyboard interrupt. Kill? (Y/N) '
                                         '(or ^c again)')
                    except KeyboardInterrupt:
                        return True
                    response = response.lower()
                    if response == 'y':
                        return True
                    elif response == 'n':
                        print('Cancelling interrupt. Starting epoch over.')
                        return False
                    else:
                        print('Unknown response')

            kill = stop_training_query()

            if kill:
                print('Training interrupted')
                exp.save(model, prefix='interrupted')
                sys.exit(0)

    logger.info('Successfully completed training')
    exp.save(model, prefix='final')
