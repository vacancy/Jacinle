#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

import jacinle.io as io
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.io import load_weights
from jactorch.utils.meta import as_float

from jaclearn.mldash import MLDashClient

logger = get_logger(__file__)

parser = JacArgumentParser(description='')
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--expr', default='default', metavar='S', help='experiment name')
parser.add_argument('--config', type='kv', nargs='*', metavar='CFG', help='extra config')
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='batch size')

# data related
# TODO(Jiayuan Mao @ 04/23): add data related arguments.
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=False, metavar='B', help='use tensorboard or not')
parser.add_argument('--debug', action='store_true', help='entering the debug mode, suppressing all logs to disk')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()
mldash = MLDashClient('dumps')

# TODO(Jiayuan Mao @ 05/03): change the filename settings.
args.series_name = 'default'
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'test-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

desc = load_source(args.desc)

# NB(Jiayuan Mao @ 02/15): compatible with the old version.
if hasattr(desc, 'configs'):
    configs = desc.configs
else:
    from jacinle.config.environ_v2 import configs

if args.config is not None:
    for c in args.config:
        c.apply(configs)

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)


def main():
    # directories
    if not args.debug:
        args.dump_dir = ensure_path(osp.join('dumps', args.series_name, args.desc_name, args.expr, args.run_name))
        args.meta_file = osp.join(args.dump_dir, 'metainfo.json')
        args.log_file = osp.join(args.dump_dir, 'log.log')
        args.meter_file = osp.join(args.dump_dir, 'meter.json')
        args.vis_dir = osp.join(args.dump_dir, 'visualizations')

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
        else:
            args.tb_dir = None

    if not args.debug:
        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

    if args.debug and args.use_tb:
        logger.warning('Disabling the tensorboard in the debug mode.')
        args.use_tb = False

    # TODO(Jiayuan Mao @ 04/23): load the dataset.
    logger.critical('Loading the dataset.')
    validation_dataset = None
    # configs.validate_dataset_compatibility(train_dataset)

    # TODO(Jiayuan Mao @ 04/23): build the model.
    logger.critical('Building the model.')
    model = desc.make_model(args)

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # from jactorch.parallel import UserScatteredJacDataParallel as JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        # TODO(Jiayuan Mao @ 04/23): disable the cudnn benchmark.
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    parent_meta_file = None
    if args.load is not None:
        raw = load_weights(model, args.load, return_raw=True)
        if raw is not None:
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))
            parent_meta_file = raw['extra']['meta_file']

    if args.use_tb:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

        logger.critical('Initializing MLDash.')
        mldash.init(
            desc_name=args.series_name + '/' + args.desc_name,
            expr_name=args.expr,
            run_name=args.run_name,
            args=args,
            highlight_args=parser,
            configs=configs,
        )
        mldash.update(metainfo_file=args.meta_file, log_file=args.log_file, meter_file=args.meter_file, tb_dir=args.tb_dir)

        if parent_meta_file is not None:
            try:
                parent_run = io.load(parent_meta_file)['args']['run_name']
                logger.critical('Setting parent run: {}.'.format(parent_run))
                mldash.update_parent(parent_run, is_master=False)
            except:
                logger.exception('Exception occurred during loading metainfo.')

    if args.embed:
        from IPython import embed; embed()

    # TODO(Jiayuan Mao @ 04/23): make the data loader.
    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    model.eval()
    with torch.no_grad():
        validate_epoch(model, validation_dataloader, meters)

    if not args.debug:
        meters.dump(args.meter_file)

    logger.critical(meters.format_simple('Test', compressed=False))


def validate_epoch(model, val_dataloader, meters):
    end = time.time()
    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            output_dict = model(feed_dict)

            # TODO(Jiayuan Mao @ 04/26): compute the monitoring values.
            monitors = as_float(output_dict['monitors'])
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/other metrics by adding n=xxx if applicable.
            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple('Test', 'val', compressed=True), refresh=False)
            pbar.update()

            end = time.time()


if __name__ == '__main__':
    main()

