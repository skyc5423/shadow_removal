'''
File: main.py
Project: sahadow_removal
File Created: 2021-08-27 15:50:12 am
Author: sangmin.lee
-----
This script ...

Reference
...
'''

import argparse, sys
from cfg import config
from cfg.config import cfg
from core.trainer import train_model, test_model


def parse_args():
    parser = argparse.ArgumentParser(description="Description for project.")

    # Mode argparse
    help_mode, choices = "Mode help description", ["train", "test"]
    parser.add_argument("--mode", help=help_mode, choices=choices, required=True, type=str)

    # Basic argparse
    help_basic = "Basic help description"
    parser.add_argument("--basic", help=help_basic, default='default arg', required=True, type=str)

    # Argparse for choices.
    help_choice, choices = "Choice argparse help description", ["choice 1", "choice 2", "choice 3", "choice 4", "choice 5"]
    parser.add_argument("--choice", help=help_choice, choices=choices, required=False, type=str)

    help_cfg = "Config file location"
    parser.add_argument("--cfg", help=help_cfg, required=True, type=str)

    help_rem = "Other configurations"
    parser.add_argument("opts", help=help_rem, default=None, nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def set_config(args):
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)
    config.assert_cfg()
    cfg.freeze()


def main():
    args = parse_args()
    set_config(args)
    if args.mode == 'train':
        train_model()
        pass
    elif args.mode == 'test':
        test_model()


if __name__ == '__main__':
    sys.argv = ['./main.py', '--mode', 'train', '--basic', 'basic', '--cfg', './cfg/config.yaml', 'OUT_DIR', './result', 'DATA_DIR', './data']
    main()
