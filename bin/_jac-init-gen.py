#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _jac-init-gen.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/31/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import sys
import tempfile
import yaml
from jacinle.logging import get_logger

logger = get_logger(__file__)


def load_system_settings(root, config, bash_file):
    if 'system' not in config or config['system'] is None:
        return
    envs = config['system'].get('envs', {})
    for k, v in envs.items():
        logger.info('Export system environment variable {} = {}.'.format(k, v))
        print('export {}={}'.format(k, v), file=bash_file)


def load_vendors(root, config, bash_file):
    if 'vendors' not in config or config['vendors'] is None:
        return

    for k, v in config['vendors'].items():
        assert 'root' in v, '"root" not found in vendor: {}.'.format(k)

        logger.info('Loading vendor: {}.'.format(k))
        print('export PYTHONPATH={}:$PYTHONPATH'.format(osp.join(root, v['root'])), file=bash_file)


def load_conda_settings(root, config, bash_file):
    if 'conda' not in config or config['conda'] is None:
        return
    target_env = config['conda'].get('env', '')
    if target_env != '':
        logger.info('Using conda env: {}.'.format(target_env))
        print("""
if [[ $CONDA_DEFAULT_ENV != "{v}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate {v}
fi
""".format(v=target_env), file=bash_file)


def load_yml_config(root, bash_file, recursive=False):
    if recursive:
        last_root = None
        while root != last_root:
            yml_filename = osp.join(root, 'jacinle.yml')
            if osp.isfile(yml_filename):
                break
            last_root = root
            root = osp.dirname(root)
    else:
        yml_filename = osp.join(root, 'jacinle.yml')

    if osp.isfile(yml_filename):
        logger.critical('Loading jacinle config: {}.'.format(osp.abspath(yml_filename)))
        with open(yml_filename) as f:
            config = yaml.safe_load(f.read())
        if config is not None:
            load_system_settings(root, config, bash_file)
            load_vendors(root, config, bash_file)
            load_conda_settings(root, config, bash_file)


def main():
    f = tempfile.NamedTemporaryFile('w', delete=False)
    load_yml_config(osp.dirname(osp.dirname(__file__)), f)
    load_yml_config(os.getcwd(), f, recursive=True)
    f.close()
    print(f.name)


if __name__ == '__main__':
    main()

