# -*- coding: utf-8 -*-
# File   : pretty.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 15/02/2018
# 
# This file is part of Jacinle.

import io as _io
import json
import functools
import collections
import xml.etree.ElementTree as et
import yaml

from jacinle.utils.meta import dict_deep_kv
from jacinle.utils.printing import stformat, kvformat

from .fs import as_file_descriptor, io_function_registry

__all__ = [
    'pretty_dump', 'pretty_load',
    'dumps_json', 'dump_json', 'loads_json', 'load_json',
    'dumps_xml', 'dump_xml', 'loads_xml', 'load_xml',
    'dumps_yaml', 'dump_yaml', 'loads_yaml', 'load_yaml',
    'dumps_txt', 'dump_txt',
    'dumps_struct', 'dump_struct',
    'dumps_kv', 'dump_kv',
    'dumps_env', 'dump_env'
]


def loads_json(value):
    return json.loads(value)


def loads_xml(value):
    return _xml2dict(et.fromstring(value))


def loads_yaml(value):
    return yaml.load(value)


def dumps_txt(value):
    assert isinstance(value, collections.Iterable), 'dump(s) txt supports only list as input.'
    with _io.StringIO() as buf:
        for v in value:
            v = str(v)
            buf.write(v)
            if v[-1] != '\n':
                buf.write('\n')
        return buf.getvalue()


def dumps_json(value):
    return json.dumps(value, sort_keys=True, indent=4, separators=(',', ': '))


def dumps_xml(value, root_node='data'):
    return _dict2xml(value, root_node=root_node)


def dumps_yaml(value):
    return yaml.dump(value, width=80, indent=4)


def dumps_struct(value):
    return stformat(value)


def dumps_kv(value):
    return kvformat(value)


def dumps_env(value):
    return '\n'.join(['{} = {}'.format(k, v) for k, v in dict_deep_kv(value)])


def _wrap_load(loads_func):
    @functools.wraps(loads_func)
    def load(file, **kwargs):
        with as_file_descriptor(file, 'r') as f:
            return loads_func(f.read(), **kwargs)

    load.__name__ = loads_func.__name__[:-1]
    load.__qualname__ = loads_func.__qualname__[:-1]
    return load


def _wrap_dump(dumps_func):
    @functools.wraps(dumps_func)
    def dump(file, obj, **kwargs):
        with as_file_descriptor(file, 'w') as f:
            return f.write(dumps_func(obj, **kwargs))

    dump.__name__ = dumps_func.__name__[:-1]
    dump.__qualname__ = dumps_func.__qualname__[:-1]
    return dump


load_json = _wrap_load(loads_json)
load_xml = _wrap_load(loads_xml)
load_yaml = _wrap_load(loads_yaml)

dump_txt = _wrap_dump(dumps_txt)
dump_json = _wrap_dump(dumps_json)
dump_xml = _wrap_dump(dumps_xml)
dump_yaml = _wrap_dump(dumps_yaml)
dump_struct = _wrap_dump(dumps_struct)
dump_kv = _wrap_dump(dumps_kv)
dump_env = _wrap_dump(dumps_env)

io_function_registry.register('pretty_load', '.json', load_json)
io_function_registry.register('pretty_load', '.xml',  load_xml)
io_function_registry.register('pretty_load', '.yaml', load_yaml)

io_function_registry.register('pretty_dump', '.txt',    dump_txt)
io_function_registry.register('pretty_dump', '.json',   dump_json)
io_function_registry.register('pretty_dump', '.xml',    dump_xml)
io_function_registry.register('pretty_dump', '.yaml',   dump_yaml)
io_function_registry.register('pretty_dump', '.struct', dump_struct)
io_function_registry.register('pretty_dump', '.kv',     dump_kv)
io_function_registry.register('pretty_dump', '.env',    dump_env)


def pretty_load(file, **kwargs):
    return io_function_registry.dispatch('pretty_load', file, **kwargs)


def pretty_dump(file, obj, **kwargs):
    return io_function_registry.dispatch('pretty_dump', file, obj, **kwargs)


def _dict2xml(d, indent=4, *, root_node=None, root_indent=0):
    """Adapted from: https://gist.github.com/reimund/5435343/"""

    indent_str = '\n' + ' ' * (indent * root_indent)

    wrap = False if root_node is None or isinstance(d, list) else True
    root = 'data' if root_node is None else root_node
    root_singular = root[:-1] if 's' == root[-1] and root_node is None else root
    xml = ''
    children = []

    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                children.append(_dict2xml(value, indent=indent, root_node=key, root_indent=root_indent+1))
            elif isinstance(value, list):
                children.append(_dict2xml(value, indent=indent, root_node=key, root_indent=root_indent))
            else:
                children.append(indent_str + ' ' * indent + '<' + key + '>' + str(value) + '</' + key + '>')
    else:
        for value in d:
            children.append(_dict2xml(value, indent=indent, root_node=root_singular, root_indent=root_indent+1))

    end_tag = '>' if len(children) > 0 else '/>'

    if wrap or isinstance(d, dict):
        xml = indent_str + '<' + root + xml + end_tag

    if len(children) > 0:
        for child in children:
            xml = xml + child

        if wrap or isinstance(d, dict):
            xml = xml + indent_str + '</' + root + '>'

    return xml


def _xml2dict(element):
    output_dict = {}
    output_dict.update(element.attrib)
    if len(output_dict) == 0 and len(element) == 0:
        return element.text  # is a leaf node

    list_elements = set()

    for c in element:
        if c.tag in output_dict and c.tag not in list_elements:
            output_dict[c.tag] = [output_dict[c.tag]]
            list_elements.add(c.tag)
        if c.tag in output_dict:
            output_dict[c.tag].append(_xml2dict(c))
        else:
            output_dict[c.tag] = _xml2dict(c)
    return output_dict
