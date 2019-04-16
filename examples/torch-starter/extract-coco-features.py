#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : extract-coco-features.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Extracing features from the MS-COCO dataset.

Examples:
jac-crun 0 extract-coco-features.py --caption /mnt/localssd1/coco/annotations/captions_train2014.json --image-root /mnt/localssd1/coco/raw/train2014 --output /mnt/localssd2/train.h5
"""

import os.path as osp
import queue
import threading

from PIL import Image

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

from torch.utils.data.dataset import Dataset

import jacinle.io as io

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.utils.tqdm import tqdm
from jactorch.cuda.copy import async_copy_to

logger = get_logger(__file__)
io.set_fs_verbose(True)

parser = JacArgumentParser()
parser.add_argument('--caption', required=True, type='checked_file', help='caption annotations (*.json)')
parser.add_argument('--image-root', required=True, type='checked_dir', help='image directory')
parser.add_argument('--output', required=True, help='output .h5 file')

parser.add_argument('--image-size', default=224, type=int, metavar='N', help='input image size')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='batch size')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()
args.output_images_json = osp.splitext(args.output)[0] + '.images.json'

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)


class COCOImageDataset(Dataset):
    def __init__(self, images, image_root, image_transform):
        self.images = images
        self.image_root = image_root
        self.image_transform = image_transform

    def __getitem__(self, index):
        info = self.images[index]

        feed_dict = GView()
        feed_dict.image_filename = info['file_name']
        if self.image_root is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image = self.image_transform(feed_dict.image)

        return feed_dict.raw()

    def __len__(self):
        return len(self.images)

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'image_filename': 'skip',
        }

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        import jactorch.models.vision.resnet as resnet
        self.resnet = resnet.resnet152(pretrained=True, incl_gap=False, num_classes=None)

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        f = self.resnet(feed_dict.image)
        output_dict = {'features': f}
        return output_dict


class AsyncWriter(object):
    def __init__(self, output_file, total_size):
        self.output_file = output_file
        self.total_size = total_size

        self.queue = queue.Queue(maxsize=5)
        self.output_dataset = None

        self.thread = threading.Thread(target=self.target)
        self.thread.start()

    def feed(self, payload):
        self.queue.put(payload)

    def join(self):
        self.queue.put(None)
        self.thread.join()

    def target(self):
        cur_idx = 0

        while True:
            payload = self.queue.get()

            if payload is None:
                break

            output_dict = payload

            if self.output_dataset is None:
                logger.info('Initializing the dataset.')
                self.output_dataset = {
                    k: self.output_file.create_dataset(k, (self.total_size, ) + v.size()[1:], dtype='float32')
                    for k, v in output_dict.items()
                }

            for k, v in output_dict.items():
                next_idx = cur_idx + v.size(0)
                self.output_dataset[k][cur_idx:next_idx] = v.cpu().numpy()

            cur_idx = next_idx


def main():
    logger.critical('Loading the dataset.')
    data = io.load(args.caption)
    # Step 1: filter out images.
    images = {c['image_id'] for c in data['annotations']}
    # Step 2: build a reverse mapping for images.
    id2image = {i['id']: i for i in data['images']}
    images = [id2image[i] for i in images]

    import torchvision.transforms as T
    image_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = COCOImageDataset(images, args.image_root, image_transform)

    logger.critical('Building the model.')

    model = FeatureExtractor()
    if args.use_gpu:
        model.cuda()
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        cudnn.benchmark = True

    model.eval()
    dataloader = dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)
    output_file = io.open_h5(args.output, 'w')
    writer = AsyncWriter(output_file, total_size=len(dataset))

    for feed_dict in tqdm(dataloader, total=len(dataloader), desc='Extracting features'):
        if args.use_gpu:
            feed_dict = async_copy_to(feed_dict, 0)

        with torch.no_grad():
            output_dict = model(feed_dict)

        writer.feed(output_dict)

    writer.join()
    output_file.close()

    io.dump(args.output_images_json, images)


if __name__ == '__main__':
    main()

