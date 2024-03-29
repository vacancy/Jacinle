#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : logging.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import logging
import functools
import sys

__all__ = ['set_output_file', 'set_logger_output_file', 'set_default_level', 'get_logger']

_g_default_level = logging.INFO
_g_all_loggers = []


def set_output_file(fout, mode='a'):
    """The the output file for all loggers.

    Args:
        fout: the output file path.
        mode: the mode to open the file.
    """
    if isinstance(fout, str):
        fout = open(fout, mode)
    JacLogFormatter.log_fout = fout


def set_logger_output_file(fout, mode='a'):
    """set the output file for all loggers. Alias to :func:`set_output_file`.

    Args:
        fout: the output file path.
        mode: the mode to open the file.
    """
    set_output_file(fout, mode=mode)


def set_default_level(level, update_existing=True):
    """Set the default logging level for all loggers.

    Args:
        level: the level to set.
        update_existing: whether to update the existing loggers.
    """
    global _g_default_level
    _g_default_level = level

    if update_existing:
        for i in _g_all_loggers:
            i.setLevel(level)


def set_logger_default_level(level, update_existing=True):
    """Set the default logging level for all loggers. Alias to :func:`set_default_level`.

    Args:
        level: the level to set.
        update_existing: whether to update the existing loggers.
    """
    set_default_level(level, update_existing=update_existing)


class JacLogFormatter(logging.Formatter):
    log_fout = None
    date_full = '[%(asctime)s %(lineno)d@%(filename)s:%(name)s] '
    date = '%(asctime)s '
    msg = '%(message)s'
    max_lines = 256

    def _color_dbg(self, msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    def _color_warn(self, msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    def _color_err(self, msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    def _color_omitted(self, msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    def _color_normal(self, msg):
        return msg

    def _color_date(self, msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, 'DBG'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, 'WRN'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, 'ERR'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super().format(record)
            nr_line = formatted.count('\n') + 1
            if nr_line >= self.max_lines:
                head, body = formatted.split('\n', 1)
                formatted = '\n'.join([
                    head,
                    'BEGIN_LONG_LOG_{}_LINES{{'.format(nr_line - 1),
                    body,
                    '}}END_LONG_LOG_{}_LINES'.format(nr_line - 1)
                ])
            self.log_fout.write(formatted)
            self.log_fout.write('\n')
            self.log_fout.flush()

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super().format(record)
        nr_line = formatted.count('\n') + 1
        if nr_line >= self.max_lines:
            lines = formatted.split('\n')
            remain = self.max_lines//2
            removed = len(lines) - remain * 2
            if removed > 0:
                mid_msg = self._color_omitted(
                    '[{} log lines omitted (would be written to output file if set_output_file() has been called;\n'
                    ' the threshold can be set at TALogFormatter.max_lines)].'.format(removed))
                formatted = '\n'.join(
                    lines[:remain] + [mid_msg] + lines[-remain:])

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt


def get_logger(name=None, formatter=JacLogFormatter):
    """Get logger with given name.

    Args:
        name: the name of the logger.
        formatter: the formatter to use.
    """

    logger = logging.getLogger(name)
    if getattr(logger, '_init_done__', None):
        return logger
    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(_g_default_level)
    logger.warning_once = functools.partial(_warning_once, logger=logger)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
    handler.setLevel(0)
    del logger.handlers[:]
    logger.addHandler(handler)
    _g_all_loggers.append(logger)
    return logger


@functools.lru_cache(128)
def _warning_once(message: str, logger):
    """Print a warning message only once.

    Args:
        logger: the logger to use.
        message: the message to print.
    """
    return logger.warning(message)


logger = get_logger('Jacinle')
"""The default logger of Jacinle."""

