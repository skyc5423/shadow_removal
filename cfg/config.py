'''
File: config.py
Project: skeleton
File Created: 2021-08-27 15:57:28 am
Author: sangmin.lee
-----
This script ...

Reference
...
'''

from yacs.config import CfgNode
import os
from iopath.common.file_io import PathManagerFactory

# instantiate global path manager for pycls
pathmgr = PathManagerFactory.get()

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C
# ---------------------------------- Model options ----------------------------------- #
_C.NUM_GPUS = 1

_C.LEARN_RATE = 0.002
_C.MOMENTUM = 0.99
_C.MAX_EPOCH = 20

_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 0
_C.DATA_LOADER.TRAIN_BATCH = 4
_C.DATA_LOADER.TEST_BATCH = 4

_C.OUT_DIR = './output'
_C.DATA_DIR = './'

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

# --------------------------------- Deprecated keys ---------------------------------- #
_C.register_deprecated_key("TMP_DEPRECATED_KEYS")


def assert_cfg():
    """Checks config values invariants."""
    assert True


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)
