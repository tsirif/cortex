'''Module for training.

'''

import logging
import pprint
import sys
import time

from . import (exp, random, viz)
from .utils import (convert_to_numpy, summarize_results, update_dict_of_lists)
from .viz import plot

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.train')


def train_epoch(model, epoch, full_eval_during_train, validate_batches=0,
                data_mode='train', use_pbar=True):
    model.train_loop(epoch, data_mode=data_mode, use_pbar=use_pbar,
                     validate_batches=0)

    if full_eval_during_train:
        return eval_epoch(model, epoch, data_mode=data_mode, use_pbar=use_pbar)

    results = summarize_results(model._all_epoch_results)
    return results


def eval_epoch(model, epoch, data_mode='train', use_pbar=True):
    model.eval_loop(epoch, data_mode=data_mode, use_pbar=use_pbar)
    results = summarize_results(model._all_epoch_results)
    return results


def test_epoch(model, epoch, data_mode='test', use_pbar=True, test_seed=None):
    train_prg_state = None
    if test_seed is not None:
        train_prg_state = random.get_prgs_state()
        random.reseed(test_seed)

    results = eval_epoch(model, epoch, data_mode=data_mode, use_pbar=use_pbar)

    model.data.reset(make_pbar=False, mode='test')
    model.data.next()
    model.visualize(auto_input=True)

    if train_prg_state is not None:
        random.set_prgs_state(train_prg_state)

    return results


def display_results(train_results, valid_results, test_results,
                    epoch, epochs, epoch_time, total_time, nl=0):
    if epochs and epoch:
        nl += 1
        print('\n\tEpoch {}/{} took {:.3f}s. Total time: {:.2f}s'
              .format(epoch, epochs, epoch_time, total_time))

    times = train_results.pop('times', None)
    if times:
        nl += 1
        time_strs = ['{}: {:.2f}'.format(k, v[0]) for k, v in times.items()]
        print('\tAvg update times:: ' + ' | '.join(time_strs))

    train_losses = train_results.pop('losses')
    test_losses = test_results.pop('losses', None)

    nl += 1
    loss_strs = ['{}: {:.2f} ({:.2f})'
                 .format(k, train_losses[k][0], train_losses[k][1])
                 for k in train_losses.keys()]
    print('\tAvg (std) train loss:: ' + ' | '.join(loss_strs))
    nl += 1
    if test_losses:
        loss_strs = ['{}: {:.2f} ({:.2f})'
                     .format(k, test_losses[k][0], test_losses[k][1])
                     for k in test_losses.keys()]
        print('\tAvg (std) test loss:: ' + ' | '.join(loss_strs))
    else:
        print('\tAvg (std) test loss:: -')

    tunables = train_results.pop('tunables', None)
    nl += 3
    print('\n\tAvg (std) train results')
    print(  '\t-----------------------')
    for k, v_train in train_results.items():
        nl += 1
        if isinstance(v_train, dict):
            print('\t{}:: '.format(k) + ' | '.join(['{}: {:.2f} ({:.2f})'
                  .format(k_, v[0], v[1]) for k_, v in v_train.items()]))
        else:
            print('\t{}: {:.2f} ({:.2f})'.format(k, v_train[0], v_train[1]))
    if tunables:
        nl += 1
        print('\ttunables:: ' + ' | '.join(['{}: {:.2f}'
              .format(k_, v) for k_, v in tunables.items()]))

    if valid_results:
        nl += 3
        print('\n\tAvg validation results')
        print(  '\t----------------------')
        for k, v_valid in valid_results.items():
            nl += 1
            if isinstance(v_valid, dict):
                print('\t{}:: '.format(k) + ' | '.join(['{}: {:.2f}'
                      .format(k_, v) for k_, v in v_valid.items()]))
            else:
                print('\t{}: {:.2f}'.format(k, v_valid))

    if test_results:
        nl += 3
        print('\n\tAvg (std) test results')
        print(  '\t----------------------')
        for k, v_test in test_results.items():
            nl += 1
            if isinstance(v_test, dict):
                print('\t{}:: '.format(k) + ' | '.join(['{}: {:.2f} ({:.2f})'
                      .format(k_, v[0], v[1]) for k_, v in v_test.items()]))
            else:
                print('\t{}: {:.2f} ({:.2f})'.format(k, v_test[0], v_test[1]))

    if epoch and epochs and epoch != epochs:
        if nl > 0:
            sys.stdout.write("\033[F" * nl)
            sys.stdout.flush()


def align_summaries(d_train, d_test):
    keys = set(d_train.keys()).union(set(d_test.keys()))
    for k in keys:
        if k in d_train and k in d_test:
            v_train = d_train[k]
            v_test = d_test[k]
            if isinstance(v_train, dict):
                max_train_len = max([len(v) for v in v_train.values()])
                max_test_len = max([len(v) for v in v_test.values()])
                max_len = max(max_train_len, max_test_len)
                for k_, v in v_train.items():
                    if len(v) < max_len:
                        v_train[k_] = (v_train[k_] + [v_train[k_][-1]] *
                                       (max_len - len(v_train[k_])))
                for k_, v in v_test.items():
                    if len(v) < max_len:
                        v_test[k_] = (v_test[k_] + [v_test[k_][-1]] *
                                      (max_len - len(v_test[k_])))
            else:
                if len(v_train) > len(v_test):
                    d_test[k] = (v_test + [v_test[-1]] *
                                 (len(v_train) - len(v_test)))
                elif len(v_test) > len(v_train):
                    d_train[k] = (v_train + [v_train[-1]] *
                                  (len(v_test) - len(v_train)))
        elif k in d_train:
            v_train = d_train[k]
            if isinstance(v_train, dict):
                max_len = max([len(v) for v in v_train.values()])
                for k_, v in v_train.items():
                    if len(v) < max_len:
                        v_train[k_] = (v_train[k_] + [v_train[k_][-1]] *
                                       (max_len - len(v_train[k_])))
        elif k in d_test:
            v_test = d_test[k]
            if isinstance(v_test, dict):
                max_len = max([len(v) for v in v_test.values()])
                for k_, v in v_test.items():
                    if len(v) < max_len:
                        v_test[k_] = v_test[k_] + [v_test[k_][-1]] * \
                            (max_len - len(v_test[k_]))


def save_best(model, train_results, best, save_on_best, save_on_lowest):
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
            print(
                '\nFound best {} (train): {}'.format(
                    save_on_best, best))
            exp.save(model, prefix='best_' + save_on_best)

        return best


def main_loop(model, epochs=500, validate_batches=0,
              archive_every=10, test_every=1, test_seed=None,
              save_on_best=None, save_on_lowest=None, save_on_highest=None,
              full_eval_during_train=False,
              train_mode='train', test_mode='test', eval_only=False,
              pbar_off=False):
    '''

    Args:
        epochs: Number of epochs.
        validate_batches: How many batches to be used in each epoch to validate
            the model in the beginning of its training loop.
        archive_every: Period of epochs for writing checkpoints.
        test_every: Period of epochs for performing tests and
            expensive visualizations.
        test_seed: If not ``None``, then model testing happens under the same
            seeding conditions every time. Training PRG state is not affected
            by this choice or by the testing.
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

    if (viz.visualizer):
        viz.visualizer.text(info, env=exp.NAME, win='info')
    total_time = 0.

    if eval_only:
        test_results = test_epoch(model, 'Testing',
                                  data_mode=test_mode, use_pbar=False)
        convert_to_numpy(test_results)

        display_results(test_results, dict(), dict(),
                        'Evaluation', None, None, None)
        exit(0)

    best = None
    if not isinstance(epochs, int):
            epochs = epochs['epochs']
    epoch = exp.INFO['epoch']
    first_epoch = epoch

    while epoch < epochs:
        try:
            logger.info('Epoch {} / {}'.format(epoch, epochs))
            start_time = time.time()

            # TRAINING
            train_results_ = train_epoch(
                model, epoch, full_eval_during_train,
                validate_batches=validate_batches,
                data_mode=train_mode, use_pbar=not(pbar_off))
            convert_to_numpy(train_results_)

            if save_on_best or save_on_highest or save_on_lowest:
                best = save_best(model, train_results_, best, save_on_best,
                                 save_on_lowest)

            # VALIDATION
            valid_results_ = train_results_.pop('validate', None)
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)
            if 'validate' not in exp.SUMMARY:
                exp.SUMMARY['validate'] = dict()
            update_dict_of_lists(exp.SUMMARY['validate'], **valid_results_)

            # TESTING
            test_results_ = dict()
            if (test_every and epoch % test_every == 0) or epoch == epochs - 1:
                if not(pbar_off):
                    sys.stdout.write("\r")
                    sys.stdout.flush()
                test_results_ = test_epoch(model, epoch, data_mode=test_mode,
                                           use_pbar=not(pbar_off),
                                           test_seed=test_seed)
                convert_to_numpy(test_results_)
                update_dict_of_lists(exp.SUMMARY['test'], **test_results_)

            # Finishing up
            align_summaries(exp.SUMMARY['train'], exp.SUMMARY['test'])
            epoch_time = time.time() - start_time
            total_time += epoch_time

            if viz.visualizer:
                plot(epoch, init=(epoch == first_epoch))
                if (test_every and epoch % test_every == 0) or \
                        epoch == epochs - 1:
                    model.viz.show()
                    model.viz.clear()

            exp.INFO['epoch'] += 1
            epoch = exp.INFO['epoch']

            display_results(train_results_, valid_results_, test_results_,
                            epoch, epochs,
                            epoch_time, total_time,
                            nl=int(not(pbar_off)))

            if (archive_every and epoch % archive_every == 0):
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
