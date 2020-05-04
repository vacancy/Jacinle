#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trainval.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/26/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar
from jaclearn.mldash import MLDashClient

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

logger = get_logger(__file__)

parser = JacArgumentParser(description='')
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--expr', default='default', metavar='S', help='experiment name')
parser.add_argument('--config', type='kv', nargs='*', metavar='CFG', help='extra config')

# training hyperparameters
# TODO(Jiayuan Mao @ 07/16): set default arguments.
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient')
parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs)')

# finetuning and snapshot
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
parser.add_argument('--save-interval', type=int, default=10, metavar='N', help='model save interval (epochs)')

# evaluation only
parser.add_argument('--evaluate', action='store_true', help='evaluate the performance of the model and exit')

# data related
# TODO(Jiayuan Mao @ 04/23): add data related arguments.
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=True, metavar='B', help='use tensorboard or not')
parser.add_argument('--debug', action='store_true', help='entering the debug mode, suppressing all logs to disk')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

# filenames
# TODO(Jiayuan Mao @ 07/16): update the series number.
args.series_name = 'default'
args.desc_name = escape_desc_name(args.desc)
if not args.evaluate:
    args.run_name = 'trainval-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
else:
    args.run_name = 'val-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

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

mldash = MLDashClient('dumps')


def main():
    # directories
    if not args.debug:
        args.dump_dir = ensure_path(osp.join('dumps', args.series_name, args.desc_name, args.expr, args.run_name))
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.vis_dir = ensure_path(osp.join(args.dump_dir, 'visualizations'))
        args.meta_file = osp.join(args.dump_dir, 'metainfo.json')
        args.log_file = osp.join(args.dump_dir, 'log.log')
        args.meter_file = osp.join(args.dump_dir, 'meter.json')

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
        else:
            args.tb_dir = None

    if not args.debug:
        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

    if args.debug and args.use_tb:
        logger.warning('Disabling the tensorboard in the debug mode.')
        args.use_tb = False
    if args.evaluate and args.use_tb:
        logger.warning('Disabling the tensorboard in the evaluation mode.')
        args.use_tb = False

    # TODO(Jiayuan Mao @ 04/23): load the dataset.
    logger.critical('Loading the dataset.')
    train_dataset = None
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
            # Set user_scattered because we will add a multi GPU wrapper to the dataloader. See below.
            model = JacDataParallel(model, device_ids=args.gpus, user_scattered=True).cuda()
        # TODO(Jiayuan Mao @ 04/23): disable the cudnn benchmark.
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        # TODO(Jiayuan Mao @ 04/23): set the default optimizer.
        trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = AdamW(trainable_parameters, args.lr, weight_decay=configs.train.weight_decay)

    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))

    trainer = TrainerEnv(model, optimizer)

    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
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
        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

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

    if args.embed:
        from IPython import embed; embed()

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    # TODO(Jiayuan Mao @ 04/23): make the data loader.
    logger.critical('Building the data loader.')
    train_dataloader = train_dataset.make_dataloader(args.batch_size, shuffle=True, drop_last=True, nr_workers=args.data_workers)
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    if args.use_gpu and args.gpu_parallel:
        from jactorch.data.dataloader import JacDataLoaderMultiGPUWrapper
        train_dataloader = JacDataLoaderMultiGPUWrapper(train_dataloader, args.gpus)
        validation_dataloader = JacDataLoaderMultiGPUWrapper(validation_dataloader, args.gpus)

    if args.evaluate:
        epoch = 0

        model.eval()
        validate_epoch(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))
        return

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        meters.reset()

        model.train()
        train_epoch(epoch, trainer, train_dataloader, meters)

        if args.validation_interval > 0 and epoch % args.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                validate_epoch(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        # TODO(Jiayuan Mao @ 02/15): config the MLDash.
        if not args.debug:
            mldash.log_metric('epoch', epoch, desc=False, expr=False)
            for key, value in meters.items():
                if key.startswith('loss') or key.startswith('validation/loss'):
                    mldash.log_metric_min(key, value.avg)
            for key, value in meters.items():
                if key.startswith('acc') or key.startswith('validation/acc'):
                    mldash.log_metric_max(key, value.avg)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))

        if not args.debug:
            if epoch % args.save_interval == 0:
                fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
                trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))


def train_epoch(epoch, trainer, train_dataloader, meters):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)

    meters.update(epoch=epoch)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)

    end = time.time()
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)

            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict)
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/monitors by adding n=xxx if applicable.
            meters.update(loss=loss)
            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            # TODO(Jiayuan Mao @ 04/23): customize the logger.
            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                {k: v for k, v in meters.val.items() if not k.startswith('validation') and k.count('/') <= 1},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)


def validate_epoch(epoch, trainer, val_dataloader, meters):
    end = time.time()
    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            output_dict, extra_info = trainer.evaluate(feed_dict)

            # TODO(Jiayuan Mao @ 04/26): compute the monitoring values.
            monitors = as_float(output_dict['monitors'])
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/other metrics by adding n=xxx if applicable.
            meters.update(monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()


if __name__ == '__main__':
    main()

