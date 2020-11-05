#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : syncable_dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/26/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import threading
from torch.utils.data.dataset import Dataset
from jactorch.data.dataloader import JacDataLoader


class SyncableDataset(Dataset):
    def __init__(self):
        """
        Initialize the global thread.

        Args:
            self: (todo): write your description
        """
        self.lock = threading.Lock()
        self.global_index = 1

    def on_recv(self, data):
        """
        Called when the global signal.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        with self.lock:
            self.global_index = data['global_index']

    def __getitem__(self, index):
        """
        Get an item at the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        time.sleep(0.6)
        with self.lock:
            return (index, self.global_index)

    def __len__(self):
        """
        Returns the number of bytes in bytes.

        Args:
            self: (todo): write your description
        """
        return 20


def main():
    """
    Main function.

    Args:
    """
    dataset = SyncableDataset()
    dataloader = JacDataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, worker_recv_fn=dataset.on_recv)

    for i, value in enumerate(dataloader):
        print(i, value)
        if i == 9:
            dataloader.send_to_worker({'global_index': 10086})


if __name__ == '__main__':
    main()
