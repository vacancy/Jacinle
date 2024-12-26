#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : html_table.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import shutil
import html
import collections
import os.path as osp
import json
import contextlib
import numpy as np
from PIL import Image
from copy import deepcopy

import jacinle.io as io
from jacinle.cli.keyboard import yes_or_no

__all__ = ['HTMLTableColumnDesc', 'HTMLTableVisualizer']


class HTMLTableColumnDesc(collections.namedtuple(
    '_HTMLTableColumnDesc', ['identifier', 'name', 'type', 'css', 'td_css'],
    defaults=(None, None)
)):
    pass


class HTMLTableVisualizer(object):
    """A helper class to generate HTML tables.

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
        self.visdir = visdir
        self.title = title
        self.allow_assets = True
        self._index_filename = None

        if self.visdir.endswith('.html'):
            self._index_filename = self.visdir
            self.visdir = osp.dirname(self.visdir)
            self.allow_assets = False

        self._index_file = None
        self._table_counter = 0
        self._row_counter = 0

        self._all_table_specs = dict()
        self._current_table_spec_stack = list()

    @property
    def _current_table_spec(self):
        return self._current_table_spec_stack[-1]

    @property
    def _current_columns(self):
        return self._all_table_specs[self._current_table_spec_stack[-1]]

    @contextlib.contextmanager
    def html(self, force_overwrite: bool = False):
        self.begin_html(force_overwrite=force_overwrite)
        yield self
        self.end_html()

    def begin_html(self, force_overwrite: bool = False):
        if self.allow_assets:
            if osp.isfile(self.visdir):
                raise FileExistsError('Visualization dir "{}" is a file.'.format(self.visdir))
            elif osp.isdir(self.visdir) and osp.isfile(self.get_index_filename()):
                if force_overwrite:
                    print('Visualization dir "{}" is not empty. Removing it.'.format(self.visdir))
                    shutil.rmtree(self.visdir)
                else:
                    if yes_or_no('Visualization dir "{}" is not empty. Do you want to overwrite?'.format(self.visdir)):
                        shutil.rmtree(self.visdir)
                    else:
                        raise FileExistsError('Visualization dir "{}" already exists.'.format(self.visdir))
            io.mkdir(self.visdir)
            io.mkdir(osp.join(self.visdir, 'assets'))
        else:
            io.mkdir(self.visdir)

        self._index_file = open(self.get_index_filename(), 'w')
        self._print('<html>')
        self._print('<head>')
        self._print('<title>{}</title>'.format(self.title))
        self._print('<style>')
        self._print('td {vertical-align:top;padding:5px}')
        self._print('pre {white-space: pre-wrap;}')
        self._print('</style>')
        self._print('</head>')
        self._print('<body>')
        self._print('<h1>{}</h1>'.format(self.title))
        self._print(self.FRAMES_JS)

    def end_html(self):
        self._print('</body>')
        self._print('</html>')
        self._index_file.close()
        self._index_file = None

    def define_table(self, columns):
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
        self.begin_table(name, columns_or_spec_id)
        yield self
        self.end_table()

    def begin_table(self, name, columns_or_spec_id):
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
        self._print('</table>')
        self._current_table_spec_stack.pop()
        self._table_counter += 1
        self._row_counter = 0

        subtable = len(self._current_table_spec_stack) > 0
        if subtable:
            self._print('</td></tr>')

    def row(self, *args, **kwargs):
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
                link, alt = self.canonicalize_link('file', obj)
                self._print('    <a class="{}" href="{}">{}</a>'.format(classname, link, alt))
            elif c.type == 'image' or c.type == 'figure':
                link, alt = self.canonicalize_link(c.type, obj, row_identifier, c.identifier)
                self._print('    <img class="{}" src="{}" alt="{}" />'.format(classname, link, alt))
            elif c.type == 'frames':
                self._print_frames(row_identifier, c, obj, classname)
            elif c.type == 'text' or c.type == 'code':
                tag = 'pre' if c.type == 'code' else 'div'
                self._print('    <{} class="{}">{}</{}>'.format(tag, classname, html.escape(str(obj)), tag))
            elif c.type == 'raw':
                self._print('    {}'.format(obj))
            else:
                raise ValueError('Unknown column type: {}.'.format(c.type))
            self._print('  </td>')
        self._print('</tr>')
        self._flush()

        self._row_counter += 1

    def _print(self, *args, **kwargs):
        assert self._index_file is not None
        print(*args, file=self._index_file, **kwargs)

    def _print_frames(self, row_identifier, column, objs, classname):
        self._print('<div class="{}" style="text-align:center;">'.format(classname))
        objs = deepcopy(objs)
        type = 'text'
        has_info = False
        for i, obj in enumerate(objs):
            if 'image' in obj:
                obj['image'] = self.canonicalize_link('image', obj['image'], row_identifier, column.identifier + '_' + str(i))
                type = 'image'
            else:
                assert 'text' in obj

            if 'info' in obj:
                has_info = True
        if type == 'text':
            self._print('    <pre class="text">{}</pre>'.format(objs[0]['text']))
        else:
            self._print('    <img class="image" src="{}" alt="{}" />'.format(*objs[0]['image']))

        if has_info:
            self._print('    <pre class="info">{}</pre>'.format('Frame #0 :: ' + objs[0]['info']))
        else:
            self._print('    <pre class="info">{}</pre>'.format('Frame #0'))

        self._print('    <button class="button prev" onclick="frameMove(this, -1)">Prev</button>')
        self._print('    <button class="button next" onclick="frameMove(this, +1)">Next</button>')
        self._print('    <input class="index" type="hidden" value="0" />')
        self._print('    <pre class="data" style="display:none">{}</pre>'.format(json.dumps(objs)))
        self._print('</div>')

    def _flush(self):
        self._index_file.flush()

    def get_index_filename(self):
        if self._index_filename is not None:
            return self._index_filename
        return osp.join(self.visdir, 'index.html')

    def get_asset_filename(self, row_identifier, col_identifier, ext):
        if not self.allow_assets:
            raise ValueError('Assets are not allowed. Specify a directory instead of a .html file.')
        table_dir = osp.join(self.visdir, 'assets', 'table{}'.format(self._table_counter))
        io.mkdir(table_dir)
        return osp.join(table_dir, '{}_{}.{}'.format(row_identifier, col_identifier, ext))

    def save_image(self, image, row_identifier, col_identifier, ext='png'):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        filename = self.get_asset_filename(row_identifier, col_identifier, ext)
        image.save(filename)
        return filename

    def save_figure(self, figure, row_identifier, col_identifier, ext='png'):
        filename = self.get_asset_filename(row_identifier, col_identifier, ext)
        figure.savefig(filename)
        return filename

    def canonicalize_link(self, filetype, obj, row_identifier=None, col_identifier=None):
        if filetype == 'file':
            assert isinstance(obj, (str, bytes))
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        elif filetype == 'image':
            if not isinstance(obj, (str, bytes)):
                obj = self.save_image(obj, row_identifier, col_identifier)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        elif filetype == 'figure':
            if not isinstance(obj, (str, bytes)):
                obj = self.save_figure(obj, row_identifier, col_identifier)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        else:
            raise ValueError('Unknown file type: {}.'.format(filetype))

    FRAMES_JS = """
<script
    src="https://code.jquery.com/jquery-3.6.0.slim.min.js"
    integrity="sha256-u7e5khyithlIdTpu22PHhENmPcRdFiHRjhAuHcs05RI="
    crossorigin="anonymous"
></script>
<script>
function frameMove(elem, offset) {
    elem = $(elem);
    window.elem = elem;
    data = JSON.parse(elem.parent().find(".data").html());
    index = parseInt(elem.parent().find(".index").val());

    nextIndex = index + offset;
    if (nextIndex < 0) nextIndex = 0;
    if (nextIndex >= data.length) nextIndex = data.length - 1;

    elem.parent().find(".index").val(nextIndex);
    if ("image" in data[nextIndex]) {
        elem.parent().find(".image").attr("src", data[nextIndex]["image"][0]).attr("alt", data[nextIndex]["image"][1]);
    } else {
        elem.parent().find(".text").html(data[nextIndex]["text"]);
    }

    if ("info" in data[nextIndex]) {
        elem.parent().find(".info").html("Frame #" + nextIndex.toString() + " :: " + data[nextIndex]["info"]);
    } else {
        elem.parent().find(".info").html("Frame #" + nextIndex.toString());
    }
}
</script>
    """

