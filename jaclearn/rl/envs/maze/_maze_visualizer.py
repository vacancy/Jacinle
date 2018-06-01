#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _maze_visualizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import base64
import numpy as np

__all__ = ['MazeVisualizer', 'render_maze']

_maze_assets = [
    b'LKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKpt',
    b'MEyFMEyFXtTaMEyFMEyFSEbQLH3SXtTaMEyFMEyFMEyFLH3SSEbQMEyFMEyFMEyFMEyFMEyFLH3SMEyFMEyFMEyFSEbQLH3SSEbQSEbQSEbQSEbQMEyFMEyFMEyFMEyFSEbQSEbQSEbQMEyFMEyFMEyFSEbQMEyFMEyFLH3SXtTaXtTaLH3SSEbQMEyFMEyFMEyFMEyFSEbQLH3SSEbQSEbQMEyFMEyFLH3SMEyFMEyFMEyFSEbQLH3SSEbQSEbQMEyFMEyFMEyFSEbQLH3SXtTaXtTaSEbQSEbQMEyFMEyFMEyFMEyFSEbQMEyFMEyFMEyFMEyFMEyFMEyFSEbQSEbQSEbQLH3SMEyFMEyFMEyFMEyFMEyFSEbQMEyFMEyFMEyFMEyFMEyFMEyFLH3SMEyFMEyFLH3SMEyFMEyFMEyFMEyFMEyFLH3SSEbQMEyFMEyFMEyFLH3SXtTaMEyFMEyFMEyFSEbQSEbQSEbQMEyFMEyFSEbQXtTaLH3SSEbQSEbQLH3SSEbQMEyFMEyFMEyFMEyFSEbQMEyFLH3SLH3SXtTaXtTaSEbQMEyFMEyFMEyFLH3SLH3SMEyFMEyFSEbQSEbQMEyFMEyFMEyFLH3SMEyFMEyFSEbQMEyFMEyFMEyFLH3SSEbQLH3SSEbQLH3SMEyFMEyFMEyFMEyFSEbQMEyFMEyFLH3SLH3SMEyFLH3SXtTaMEyFMEyFMEyFSEbQMEyFMEyFLH3SLH3SLH3SLH3SSEbQSEbQMEyFSEbQSEbQMEyFMEyFMEyFMEyFMEyFSEbQLH3SMEyFMEyFMEyFSEbQLH3SMEyFMEyFMEyFSEbQMEyFMEyFMEyFMEyFMEyFLH3SMEyFMEyFMEyFMEyFMEyFSEbQMEyFMEyFMEyFSEbQSEbQMEyFMEyFMEyFMEyFSEbQMEyFMEyFMEyFMEyFMEyFMEyFSEbQXtTaLH3SMEyFSEbQLH3SXtTaLH3SLH3SXtTaMEyFMEyFMEyFMEyFMEyFSEbQXtTaMEyFMEyF',
    b'////////////AAAAAAAAHAwUHAwUHAwUHAwUHAwUHAwUAAAAAAAA////////////////////////AAAAHAwUMEyFLH3SLH3SLH3SLH3SMEyFHAwUAAAA////////////////////////HAwUMEyFLH3SMEyFLH3SXtTaMEyFMEyFMEyFHAwU////////////////////////HAwUXtTaMEyFXtTaXtTaLH3SNCRELH3SMEyFHAwU////////////////////////HAwUMEyFLH3SNCREMEyFNCRELH3SNCREMEyFHAwU////////////////////////HAwUMEyFNCREHAwUXtTaXtTaHAwULH3SNCREHAwU////////////////////////AAAAHAwUXtTaHAwU1u7e1u7eHAwUXtTaHAwUAAAA////////////////////////////////HAwUmarSmarSmarSmarSHAwU////////////////////////////////////HAwUzn1ZMEyFLH3SLH3SMEyFzn1ZHAwU////////////////////////////HAwUzn1ZbTQwzn1Zzn1Zzn1Zzn1ZbTQwzn1ZHAwU////////////////////////HAwUmarSHAwUzn1Zzn1Zzn1Zzn1ZHAwUmarSHAwU////////////////////////////HAwUHAwUSEbQNCRENCRESEbQHAwUHAwU////////////////////////////////////HAwUSEbQNCRENCRESEbQHAwU////////////////////////////////YXF1YXF1HAwULH3SNCRENCRELH3SHAwUYXF1YXF1////////////////////////YXF1YXF1YXF1HAwUHAwUHAwUHAwUYXF1YXF1YXF1////////////////////////////YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1////////////////',
    b'LKptLKptLKptLKptLKptLKptLKptLKptLKptXtTaLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptXtTa1u7eXtTaLKptLKptLKptLKptLKptLKpt1u7eLKptLKptLKptLKptLKptLKptLKptXtTaLKptLKptLKptLKptLKptLKptLKptLKptJGU0LKptLKptLKptLKptXtTaLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptJGU0LKptLKptLKptXtTa1u7eXtTaXtTa1u7eXtTaLKptLKptLKpt1u7eLKptLKptLKptLKptLKptLKptLKptXtTaLKpt1u7eLH3S1u7eLKptLKptLKptJGU0LKptXtTa1u7eXtTaLKptLKptJGU0LKptLKptXtTa1u7eXtTaLKptXtTaLKptLKptLKpt1u7eLH3S1u7eLKptLKptJGU0LKptLKptLKptJGU0LKptXtTa1u7eXtTaLKptLKptXtTa1u7eXtTaLKptLKptLKptLKptJGU0JGU0JGU0LKptLKptXtTaLKptLKptLKptLKptJGU0LKptLKptXtTaLKptLKptLKptJGU0LKptLKptLKptJGU0JGU0LKptLKptJGU0JGU0LKptXtTa1u7eXtTaLKptLKptLKptLKptXtTaLKptJGU0LKptLKptLKptLKptJGU0LKptLKptXtTaJGU0LKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptJGU0LKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptXtTa1u7eXtTaLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptLKptJGU0JGU01u7eLH3S1u7eLKptLKptLKpt1u7eLKptLKptLKptLKptLKptLKptLKptLKptJGU0XtTa1u7eXtTa',
    b'YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1JGU0YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1TkpOTkpOTkpOTkpOJGU0TkpOTkpOYXF1JGU0JGU0JGU0YXF1TkpOTkpOTkpOYXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1JGU0YXF1YXF1YXF1TkpOTkpOYXF1TkpOTkpOTkpOTkpOJGU0TkpOTkpOYXF1JGU0JGU0TkpOTkpOYXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1JGU0YXF1YXF1YXF1JGU0JGU0YXF1YXF1JGU0YXF1YXF1YXF1YXF1TkpOYXF1TkpOJGU0TkpOTkpOJGU0TkpOTkpOJGU0TkpOTkpOTkpOTkpOYXF1TkpOTkpOTkpOYXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1YXF1YXF1YXF1JGU0YXF1JGU0JGU0YXF1YXF1YXF1YXF1TkpOYXF1YXF1YXF1YXF1JGU0YXF1TkpOTkpOYXF1TkpOTkpOTkpOTkpOJGU0TkpOTkpOTkpOYXF1JGU0JGU0JGU0'
]
_maze_colors = [(255, 255, 255), (0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
_maze_psize = 16


class MazeVisualizer(object):
    _maze_colors = None
    _maze_assets = None

    def _load_settings(self):
        if self._maze_assets is not None:
            return

        self._maze_colors = _maze_colors
        self._maze_assets = [
                np.fromstring(base64.b64decode(p), dtype='uint8').reshape((_maze_psize, _maze_psize, 3))
                for p in _maze_assets]

    def translate(self, p):
        self._load_settings()

        p = tuple(map(int, p))
        for i, c in enumerate(self._maze_colors):
            if p == c:
                return i, self._maze_assets[i]

        raise ValueError('Unknown primitive color')

    def render(self, m):
        h, w = m.shape[:2]
        ps = _maze_psize
        viz = np.zeros((h * ps, w * ps, 3), dtype='uint8')
        for i in range(h):
            for j in range(w):
                vid, v  = self.translate(m[i, j])
                if vid == 2:
                    v = self._overlap_color(v, self._maze_assets[0])
                viz[i*ps:i*ps+ps, j*ps:j*ps+ps, :] = v[:, :, :3]
        return viz

    @staticmethod
    def _overlap_color(main, bg):
        mask = (main != (255, 255, 255)).astype('float32')
        return main * mask + (1 - mask) * bg


_visualizer = MazeVisualizer()
render_maze = _visualizer.render
