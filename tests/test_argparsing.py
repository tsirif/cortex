from cortex._lib import (config, setup_experiment, exp)
from cortex.built_ins.models.classifier import ImageClassification
from cortex._lib.parsing import update_args
from cortex._lib.utils import update_nested_dict


def test_command_override_static(args):
    """

    Args:
        args(@pytest.fixture): Namespace

    Asserts: True if passing a command line arg, the exp.ARGS is
             changing the value from default for the command line
             one.

    """
    expected_type = 'resnet'
    args.__dict__['classifier_type'] = expected_type
    classifier_defaults = ImageClassification()
    config.set_config()
    # NOTE: exp.ARGS is being populated inside setup_experiment() call
    classifier_defaults = setup_experiment(
        args, model=classifier_defaults, testmode=True)
    assert exp.ARGS['model']['classifier_type'] == expected_type


def test_static_override_parameters(args, classifier_modified):
    """

    Args:
        args(@pytest.fixture): Namespace
        classifier_modified(@pytest.fixture): ClassifierModified

    Asserts: True if default attribute is overriding
             parameters values.

    """
    expected_type = 'convnet'
    config.set_config()
    classifier_modified = setup_experiment(
        args, model=classifier_modified, testmode=True)
    assert exp.ARGS['model']['classifier_type'] == expected_type


def test_update_nested_dict(args, classifier_modified):
    """

    Args:
        args(@pytest.fixture): Namespace
        classifier_modified(@pytest.fixture): ClassifierModified

    Asserts: True if a dict. arg. is being updated to a
             nested dict. (not overridden).

    """
    expected_classifier_args_before_update = {'dropout': 0.2}
    expected_classifier_args_after_update = {'dropout': 0.2, 'dim_h': 100}
    args_for_update = {
        'data': {
            'batch_size': 122
        },
        'model': {
            'classifier_args': {
                'dropout': 0.2,
                'dim_h': 100
            }
        }
    }
    config.set_config()
    classifier_modified = setup_experiment(
        args, model=classifier_modified, testmode=True)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_before_update
    update_nested_dict(exp.ARGS, args_for_update, strict=True)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_after_update
    assert exp.ARGS['data']['batch_size'] == 122


def test_update_args(args, classifier_modified):
    """

    Args:
        args(@pytest.fixture): Namespace
        classifier_modified(@pytest.fixture): ClassifierModified

    Asserts: True if exp.ARGS is updated adequately.

    """
    expected_classifier_args_before_update = {'dropout': 0.2}
    expected_classifier_args_after_update = {'dropout': 0.1}
    args_for_update = {
        'data': {
            'batch_size': 128
        },
        'model': {
            'classifier_args': {
                'dropout': 0.1
            }
        }
    }
    config.set_config()
    classifier_modified = setup_experiment(
        args, model=classifier_modified, testmode=True)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_before_update
    update_args(args_for_update, exp.ARGS)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_after_update
    assert exp.ARGS['data']['batch_size'] == 128
