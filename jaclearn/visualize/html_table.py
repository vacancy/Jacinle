#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : html_table.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import shutil
import collections
import os.path as osp
import contextlib

import jacinle.io as io
from jacinle.cli.keyboard import yes_or_no

__all__ = ['HTMLTableColumnDesc', 'HTMLTableVisualizer']


class HTMLTableColumnDesc(collections.namedtuple(
    '_HTMLTableColumnDesc', ['identifier', 'name', 'type', 'css', 'td_css'],
    defaults=(None, None)
)):
    pass


class HTMLTableVisualizer(object):
    """
    Example:
        >>> vis = HTMLTableVisualizer('<some_dir>', 'Visualization')
        >>> with vis.html():
        >>>     with vis.table('Table Name', [
        >>>         HTMLTableColumnDesc('column1', 'Image', 'image', {'width': '120px'}),
        >>>         HTMLTableColumnDesc('column2', 'Result', 'figure' {}),
        >>>         HTMLTableColumnDesc('column3', 'Supervision', 'text' {}),
        >>>         HTMLTableColumnDesc('column4', 'Prediction', 'code' {'font-size': '12px'})
        >>>     ]):
        >>>         vis.row(...)
    """

    def __init__(self, visdir, title):
        """
        Initialize table.

        Args:
            self: (todo): write your description
            visdir: (str): write your description
            title: (str): write your description
        """
        self.visdir = visdir
        self.title = title

        self._index_file = None
        self._table_counter = 0
        self._row_counter = 0

        self._all_table_specs = dict()
        self._current_table_spec_stack = list()

    @property
    def _current_table_spec(self):
        """
        Return the current table spec.

        Args:
            self: (todo): write your description
        """
        return self._current_table_spec_stack[-1]

    @property
    def _current_columns(self):
        """
        Return the current columns.

        Args:
            self: (todo): write your description
        """
        return self._all_table_specs[self._current_table_spec_stack[-1]]

    @contextlib.contextmanager
    def html(self):
        """
        Generate the html.

        Args:
            self: (todo): write your description
        """
        self.begin_html()
        yield self
        self.end_html()

    def begin_html(self):
        """
        Create the html file.

        Args:
            self: (todo): write your description
        """
        if osp.isfile(self.visdir):
            raise FileExistsError('Visualization dir "{}" is a file.'.format(self.visdir))
        elif osp.isdir(self.visdir) and osp.isfile(self.get_index_filename()):
            if yes_or_no('Visualization dir "{}" is not empty. Do you want to overwrite?'.format(self.visdir)):
                shutil.rmtree(self.visdir)
            else:
                raise FileExistsError('Visualization dir "{}" already exists.'.format(self.visdir))

        io.mkdir(self.visdir)
        io.mkdir(osp.join(self.visdir, 'assets'))
        self._index_file = open(self.get_index_filename(), 'w')
        self._print('<html>')
        self._print('<head>')
        self._print('<title>{}</title>'.format(self.title))
        self._print('<style>')
        self._print('td {vertical-align:top;padding:5px}')
        self._print('</style>')
        self._print('</head>')
        self._print('<body>')
        self._print('<h1>{}</h1>'.format(self.title))

    def end_html(self):
        """
        Displays the index.

        Args:
            self: (todo): write your description
        """
        self._print('</body>')
        self._print('</html>')
        self._index_file.close()
        self._index_file = None

    def define_table(self, columns):
        """
        Define the columns dictionary.

        Args:
            self: (todo): write your description
            columns: (list): write your description
        """
        spec_id = len(self._all_table_specs)
        self._print('<style>')
        for c in columns:
            css = {} if c.css is None else c.css
            self._print('.table{}_column_{}'.format(spec_id, c.identifier), '{', ';'.join([k + ':' + v for k, v in css.items()]), '}')
            css = {} if c.td_css is None else c.td_css
            self._print('.table{}_td_{}'.format(spec_id, c.identifier), '{', ';'.join([k + ':' + v for k, v in css.items()]), '}')
        self._print('</style>')
        self._all_table_specs[spec_id] = columns
        return spec_id

    @contextlib.contextmanager
    def table(self, name, columns_or_spec_id):
        """
        Yield a table.

        Args:
            self: (todo): write your description
            name: (str): write your description
            columns_or_spec_id: (str): write your description
        """
        self.begin_table(name, columns_or_spec_id)
        yield self
        self.end_table()

    def begin_table(self, name, columns_or_spec_id):
        """
        Begin a new table.

        Args:
            self: (todo): write your description
            name: (str): write your description
            columns_or_spec_id: (str): write your description
        """
        subtable = len(self._current_table_spec_stack) > 0
        if subtable:
            self._print('<tr><td style="padding-left:50px" colspan="{}">'.format(len(self._current_columns)))

        if isinstance(columns_or_spec_id, int):
            self._current_table_spec_stack.append(columns_or_spec_id)
        else:
            self._current_table_spec_stack.append(self.define_table(columns_or_spec_id))

        if name is not None:
            self._print('<h3>{}</h3>'.format(name))

        self._print('<table>')
        self._print('<tr>')
        for c in self._current_columns:
            self._print('  <td><b>{}</b></td>'.format(c.name))
        self._print('</tr>')

    def end_table(self):
        """
        End the current stack.

        Args:
            self: (todo): write your description
        """
        self._print('</table>')
        self._current_table_spec_stack.pop()
        self._table_counter += 1
        self._row_counter = 0

        subtable = len(self._current_table_spec_stack) > 0
        if subtable:
            self._print('</td></tr>')

    def row(self, *args, **kwargs):
        """
        Display the contents of the class

        Args:
            self: (todo): write your description
        """
        assert len(self._current_table_spec_stack) > 0

        if len(args) > 0:
            assert len(kwargs) == 0 and len(args) == len(self._current_columns)
            for c, a in zip(self._current_columns, args):
                kwargs[c.identifier] = a

        row_identifier = kwargs.pop('row_identifier', 'row{:06d}'.format(self._row_counter))

        self._print('<tr>')
        for c in self._current_columns:
            obj = kwargs[c.identifier]
            classname = 'table{}_td_{}'.format(self._current_table_spec, c.identifier)
            self._print('  <td class="{}">'.format(classname))
            classname = 'table{}_column_{}'.format(self._current_table_spec, c.identifier)
            if obj is None:
                pass
            elif c.type == 'file':
                link, alt = self.canonize_link('file', obj)
                self._print('    <a class="{}" href="{}">{}</a>'.format(classname, link, alt))
            elif c.type == 'image' or c.type == 'figure':
                link, alt = self.canonize_link(c.type, obj, row_identifier, c.identifier)
                self._print('    <img class="{}" src="{}" alt="{}" />'.format(classname, link, alt))
            elif c.type == 'text' or c.type == 'code':
                tag = 'pre' if c.type == 'code' else 'div'
                self._print('    <{} class="{}">{}</{}>'.format(tag, classname, obj, tag))
            elif c.type == 'raw':
                self._print('    {}'.format(obj))
            else:
                raise ValueError('Unknown column type: {}.'.format(c.type))
            self._print('  </td>')
        self._print('</tr>')
        self._flush()

        self._row_counter += 1

    def _print(self, *args, **kwargs):
        """
        Print the index of the index.

        Args:
            self: (todo): write your description
        """
        assert self._index_file is not None
        print(*args, file=self._index_file, **kwargs)

    def _flush(self):
        """
        Flush the index file.

        Args:
            self: (todo): write your description
        """
        self._index_file.flush()

    def get_index_filename(self):
        """
        Return the index of the index file.

        Args:
            self: (todo): write your description
        """
        return osp.join(self.visdir, 'index.html')

    def get_asset_filename(self, row_identifier, col_identifier, ext):
        """
        Generate an asset filename.

        Args:
            self: (str): write your description
            row_identifier: (str): write your description
            col_identifier: (str): write your description
            ext: (str): write your description
        """
        table_dir = osp.join(self.visdir, 'assets', 'table{}'.format(self._table_counter))
        io.mkdir(table_dir)
        return osp.join(table_dir, '{}_{}.{}'.format(row_identifier, col_identifier, ext))

    def save_image(self, image, row_identifier, col_identifier, ext='png'):
        """
        Save an image.

        Args:
            self: (todo): write your description
            image: (array): write your description
            row_identifier: (str): write your description
            col_identifier: (str): write your description
            ext: (str): write your description
        """
        filename = self.get_asset_filename(row_identifier, col_identifier, ext)
        image.save(filename)
        return filename

    def save_figure(self, figure, row_identifier, col_identifier, ext='png'):
        """
        Save figure as png file.

        Args:
            self: (todo): write your description
            figure: (todo): write your description
            row_identifier: (str): write your description
            col_identifier: (str): write your description
            ext: (str): write your description
        """
        filename = self.get_asset_filename(row_identifier, col_identifier, ext)
        figure.savefig(filename)
        return filename

    def canonize_link(self, filetype, obj, row_identifier=None, col_identifier=None):
        """
        Return link to an object.

        Args:
            self: (todo): write your description
            filetype: (str): write your description
            obj: (todo): write your description
            row_identifier: (str): write your description
            col_identifier: (str): write your description
        """
        if filetype == 'file':
            assert isinstance(obj, six.string_types)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        elif filetype == 'image':
            if not isinstance(obj, six.string_types):
                obj = self.save_image(obj, row_identifier, col_identifier)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        elif filetype == 'figure':
            if not isinstance(obj, six.string_types):
                obj = self.save_figure(obj, row_identifier, col_identifier)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        else:
            raise ValueError('Unknown file type: {}.'.format(filetype))

