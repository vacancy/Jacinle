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

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.io import load_weights
from jactorch.utils.meta import as_float

logger = get_logger(__file__)

parser = JacArgumentParser(description='')
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')

# data related
# TODO(Jiayuan Mao @ 04/23): add data related arguments.
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--debug', action='store_true', help='entering the debug mode, suppressing all logs to disk')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

# filenames
args.series_name = 'default'
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'test-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

desc = load_source(args.desc)
configs = desc.configs

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
        args.dump_dir = ensure_path(osp.join('dumps', args.series_name, args.desc_name))
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
        args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')
        args.log_file = osp.join(args.meta_dir, args.run_name + '.log')
        args.meter_file = osp.join(args.meta_dir, args.run_name + '.meter.json')

    if not args.debug:
        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))
    else:
        if args.use_tb:
            logger.warning('Disabling the tensorboard in the debug mode.'.format(args.meta_file))
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

    if load_weights(model, args.load):
        logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))

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

    if args.embed:
        from IPython import embed; embed()

    # TODO(Jiayuan Mao @ 04/23): make the data loader.
    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    model.eval()
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
            monitors = as_float(output['monitors'])
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/other metrics by adding n=xxx if applicable.
            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple('Test', 'val', compressed=True))
            pbar.update()

            end = time.time()


if __name__ == '__main__':
    main()

