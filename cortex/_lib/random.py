# -*- coding: utf-8 -*-
"""Module for extracting and setting states for the possible PRGs.

Disclaimer: This module's "private" functions are copied and
            adapted from Nauka project.
Original Code Repo: https://github.com/obilaniu/Nauka
Original Code Copyright: Copyright (c) 2018 Olexa Bilaniuk
Original Code License: MIT License

"""
from contextlib import contextmanager
from functools import partial

import hashlib
import numpy as np
import random
import torch

from . import exp

__author__ = 'Tsirigotis Christos'
__author_email__ = 'tsirif@gmail.com'


def _toBytesUTF8(x, errors="strict"):
    return x.encode("utf-8", errors=errors) if isinstance(x, str) else x


def _pbkdf2(dkLen, password, salt="", rounds=1, hash="sha256"):
    password = _toBytesUTF8(password)
    salt = _toBytesUTF8(salt)
    return hashlib.pbkdf2_hmac(hash, password, salt, rounds, dkLen)

################################################################################
#                                  Numpy PRG                                   #
################################################################################


def _getNpRandomStateFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    uint32le = np.dtype(np.uint32).newbyteorder("<")
    buf = _pbkdf2(624 * 4, password, salt, rounds, hash)
    buf = np.frombuffer(buf, dtype=uint32le).copy("C")
    return ("MT19937", buf, 624, 0, 0.0)


def _seedNpRandomFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    npRandomState = _getNpRandomStateFromPBKDF2(password, salt, rounds, hash)
    np.random.set_state(npRandomState)
    return npRandomState

################################################################################
#                               Python Math PRG                                #
################################################################################


def _getRandomStateFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    npRandomState = _getNpRandomStateFromPBKDF2(password, salt, rounds, hash)
    twisterState = tuple(npRandomState[1].tolist()) + (624,)
    return (3, twisterState, None)


def _seedRandomFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    randomState = _getRandomStateFromPBKDF2(password, salt, rounds, hash)
    random.setstate(randomState)
    return randomState


################################################################################
#                                  Torch PRG                                   #
################################################################################


def _getIntFromPBKDF2(nbits, password, salt="",
                      rounds=1, hash="sha256", signed=False):
    nbits = int(nbits)
    assert nbits % 8 == 0
    dkLen = nbits // 8
    buf = _pbkdf2(dkLen, password, salt, rounds, hash)
    return int.from_bytes(buf, "little", signed=signed)


def _getTorchRandomSeedFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    return _getIntFromPBKDF2(64, password, salt, rounds, hash, signed=True)


def _seedTorchRandomFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    seed = _getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
    torch.random.manual_seed(seed)
    return seed


def _seedTorchCudaRandomFromPBKDF2(password, salt="", rounds=1, hash="sha256",
                                   use_all=False):
    seed = _getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
    if use_all:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    return seed


def _setTorchCudaRandomFromState(state, use_all=False):
    if not torch.cuda.is_available() or str(exp.DEVICE) == 'cpu':
        return

    if use_all:
        torch.cuda.set_rng_state_all(state)
        return

    # TODO if distributed, broadcast and select (scatter) states
    torch.cuda.set_rng_state(state, device=exp.DEVICE)


def _getTorchCudaRandomState(use_all=False):
    if not torch.cuda.is_available() or str(exp.DEVICE) == 'cpu':
        return

    if use_all:
        return torch.cuda.get_rng_state_all()

    # TODO if distributed, gather states
    return torch.cuda.get_rng_state(device=exp.DEVICE)


PRGS = ("random", "numpy.random", "torch.random", "torch.cuda.random")
SEEDERS = {
    "random": _seedRandomFromPBKDF2,
    "numpy.random": _seedNpRandomFromPBKDF2,
    "torch.random": _seedTorchRandomFromPBKDF2,
    "torch.cuda": _seedTorchCudaRandomFromPBKDF2,
    }
SETTERS = {
    "random": random.setstate,
    "numpy.random": np.random.set_state,
    "torch.random": torch.random.set_rng_state,
    "torch.cuda": _setTorchCudaRandomFromState,
    }
GETTERS = {
    "random": random.getstate,
    "numpy.random": np.random.get_state,
    "torch.random": torch.random.get_rng_state,
    "torch.cuda": _getTorchCudaRandomState,
    }


def init(use_all_cuda=False):
    """Initialize module by providing info about device usage."""
    if use_all_cuda is True:
        SEEDERS["torch.cuda"] = partial(_seedTorchCudaRandomFromPBKDF2, use_all=True)
        SETTERS["torch.cuda"] = partial(_setTorchCudaRandomFromState, use_all=True)
        GETTERS["torch.cuda"] = partial(_getTorchCudaRandomState, use_all=True)


def reseed(seed, prgs=PRGS):
    """Seed PRNGs for reproducibility at beginning of interval."""
    if not isinstance(prgs, tuple):
        prgs = tuple(prgs, )
    password = "Seed: {}".format(seed)
    for prg in prgs:
        seeder = SEEDERS[prg]
        seeder(password, salt=prg)


def set_prgs_state(state, seed=None, strict=False, prgs=PRGS):
    """Set `state` for the PRGs possibly used.

    If `state` does not contain information about a certain PRG used,
    then that PRG is reseeded with `seed`, if it not ``None``.

    :param state: Dictionary with PRG's internal states.

    .. note:: Not exactly transferable to other setups, which may have
       different number of cuda devices available.

       For safe determinism, it is needed that an experiment is resumed on the
       same machine and using the same primary device.

    """
    for prg in prgs:
        if prg in state:
            setter = SETTERS[prg]
            setter(state[prg])
        else:
            if not strict:
                if seed is not None:
                    reseed(seed, prg)
            else:
                raise RuntimeError("State could not be set for PRG: {}"
                                   .format(prg))


def get_prgs_state(prgs=PRGS):
    """Extract PRGs states for checkpointing.

    :returns: a dictionary with PRG's current internal states

    .. note:: Not exactly transferable to other setups, which may have
       different number of cuda devices available.

       For safe determinism, it is needed that an experiment is resumed on the
       same machine and using the same primary device.

    """
    state = dict()
    for prg in prgs:
        getter = GETTERS[prg]
        state[prg] = getter()
    return state


@contextmanager
def fresh_prgs(seed=None, prgs=PRGS):
    prgs_state = None
    if seed is not None:
        prgs_state = get_prgs_state(prgs=prgs)
        reseed(seed, prgs=prgs)
    yield
    if prgs_state is not None:
        set_prgs_state(prgs_state, strict=True, prgs=prgs)
