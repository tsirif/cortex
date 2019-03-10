'''Main file for running experiments.

'''

import logging
import pprint

from cortex._lib import (config, data, exp, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


logger = logging.getLogger('cortex')


def run(model=None):
    '''Main function.

    '''
    # Parse the command-line arguments

    try:
        args = setup_cortex(model=model)
        if args.command == 'setup':
            # Performs setup only.
            config.setup()
            exit(0)

        config.set_config()
        print_section('EXPERIMENT')
        model, reload_nets, reload_optims = setup_experiment(args, model=model)
        print_section('DATA')
        data.setup(**exp.ARGS['data'])
        print_section('MODEL')
        model.build()
        model.reload_nets(reload_nets)
        print_section('OPTIMIZER')
        optimizer.setup(model, reload_optims,
                        **exp.ARGS['optimizer'])

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    print_section('RUNNING')
    train.main_loop(model, **exp.ARGS['train'])
