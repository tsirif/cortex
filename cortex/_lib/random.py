# -*- coding: utf-8 -*-
"""Module for extracting and setting states for the possible PRGs.

Disclaimer: This module's "private" functions are copied and
            adapted from Nauka project.
Original Code Repo: https://github.com/obilaniu/Nauka
Original Code Copyright: Copyright (c) 2018 Olexa Bilaniuk
Original Code License: MIT License
Module Author: Christos Tsirigotis (tsirif <at> gmail <dot> com)

"""

import hashlib
import numpy as np
import random
import torch

from . import exp


def _toBytesUTF8(x, errors="strict"):
    return x.encode("utf-8", errors=errors) if isinstance(x, str) else x


def _pbkdf2(dkLen, password, salt="", rounds=1, hash="sha256"):
    password = _toBytesUTF8(password)
    salt = _toBytesUTF8(salt)
    return hashlib.pbkdf2_hmac(hash, password, salt, rounds, dkLen)


def _getIntFromPBKDF2(nbits, password, salt="",
                      rounds=1, hash="sha256", signed=False):
    nbits = int(nbits)
    assert nbits % 8 == 0
    dkLen = nbits // 8
    buf = _pbkdf2(dkLen, password, salt, rounds, hash)
    return int.from_bytes(buf, "little", signed=signed)


def _getNpRandomStateFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    uint32le = np.dtype(np.uint32).newbyteorder("<")
    buf = _pbkdf2(624 * 4, password, salt, rounds, hash)
    buf = np.frombuffer(buf, dtype=uint32le).copy("C")
    return ("MT19937", buf, 624, 0, 0.0)


def _seedNpRandomFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    npRandomState = _getNpRandomStateFromPBKDF2(password, salt, rounds, hash)
    np.random.set_state(npRandomState)
    return npRandomState


def _getRandomStateFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    npRandomState = _getNpRandomStateFromPBKDF2(password, salt, rounds, hash)
    twisterState = tuple(npRandomState[1].tolist()) + (624,)
    return (3, twisterState, None)


def _seedRandomFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    randomState = _getRandomStateFromPBKDF2(password, salt, rounds, hash)
    random.setstate(randomState)
    return randomState


def _getTorchRandomSeedFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    return _getIntFromPBKDF2(64, password, salt, rounds, hash, signed=True)


def _seedTorchRandomManualFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    seed = _getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
    torch.random.manual_seed(seed)
    return seed


def _seedTorchCudaManualFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    seed = _getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
    torch.cuda.manual_seed(seed)
    return seed


def _seedTorchCudaManualAllFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
    seed = _getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
    torch.cuda.manual_seed_all(seed)
    return seed


# PRG name: (seeder, setter, getter)
PRGS = {
    "random": (_seedRandomFromPBKDF2,
               random.setstate, random.getstate),
    "numpy.random": (_seedNpRandomFromPBKDF2,
                     np.random.set_state, np.random.get_state),
    "torch.random": (_seedTorchRandomManualFromPBKDF2,
                     torch.random.set_rng_state, torch.random.get_rng_state),
    "torch.cuda": (_seedTorchCudaManualAllFromPBKDF2,
                   torch.cuda.set_rng_state, torch.cuda.get_rng_state),
    }
PRGS_NAMES = tuple(PRGS.keys())


def reseed(seed, prgs=PRGS_NAMES):
    """Seed PRNGs for reproducibility at beginning of interval."""
    if not isinstance(prgs, tuple):
        prgs = tuple(prgs, )
    password = "Seed: {}".format(seed)
    for prg in prgs:
        seeder = PRGS[prg][0]
        seeder(password, salt=prg)


def set_prgs_state(state, seed=None):
    """Set `state` for the PRGs possibly used.

    If `state` does not contain information about a certain PRG used,
    then that PRG is reseeded with `seed`, if it not ``None``.

    :param state: Dictionary with PRG's internal states.

    .. note:: Not exactly transferable to other setups, which may have
       different number of cuda devices available.

       For safe determinism, it is needed that an experiment is resumed on the
       same machine and using the same primary device.

    """
    for prg in PRGS:
        if prg in state:
            setter = PRGS[prg][1]
            if prg != "torch.cuda":
                setter(state[prg])
            elif torch.cuda.is_available() and str(exp.DEVICE) != 'cpu':
                for devnum in range(torch.cuda.device_count()):
                    setter(state[prg][devnum], device=devnum)
        elif prg != "torch.cuda" or \
                (torch.cuda.is_available() and str(exp.DEVICE) != 'cpu'):
            if seed is not None:
                reseed(seed, prg)
            else:
                raise RuntimeError("State could not be set for PRG: {}"
                                   .format(prg))


def get_prgs_state():
    """Extract PRGs states for checkpointing.

    :returns: a dictionary with PRG's current internal states

    .. note:: Not exactly transferable to other setups, which may have
       different number of cuda devices available.

       For safe determinism, it is needed that an experiment is resumed on the
       same machine and using the same primary device.

    """
    state = dict()
    for prg in PRGS:
        getter = PRGS[prg][2]
        if prg != "torch.cuda":
            state[prg] = getter()
        elif torch.cuda.is_available() and str(exp.DEVICE) != 'cpu':
            state[prg] = dict()
            for devnum in range(torch.cuda.device_count()):
                state[prg][devnum] = getter(device=devnum)

    return state
