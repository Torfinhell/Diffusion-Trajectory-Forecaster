import hashlib
from copy import deepcopy

from omegaconf import OmegaConf


def process_hparams(hparams, print_hparams=True):
    if hparams.logging == "online":
        hparams.show = False

    """Create HParam ID for saving and loading checkpoints"""
    hashable_config = deepcopy(hparams)
    id = hashlib.sha1(
        repr(sorted(hashable_config.__dict__.items())).encode()
    ).hexdigest()
    hparams.hparams_id = id
    if print_hparams:
        print(OmegaConf.to_container(hparams, resolve=True))
    return hparams
