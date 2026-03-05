import hashlib
from copy import deepcopy

from omegaconf import OmegaConf


def process_hparams(hparams, print_hparams=True):
    if hparams.trainer.logging == "online":
        hparams.show = False

    """Create HParam ID for saving and loading checkpoints"""
    hashable_config = deepcopy(hparams)
    id = hashlib.sha1(
        repr(sorted(hashable_config.__dict__.items())).encode()
    ).hexdigest()
    OmegaConf.set_struct(hparams, False)
    hparams.hparams_id = id
    OmegaConf.set_struct(hparams, True)
    if print_hparams:
        print(OmegaConf.to_container(hparams, resolve=True))
    return hparams
