#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pretty.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/15/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import io as _io
import json
import functools
import collections
import xml.etree.ElementTree as et
import yaml
import six
import inspect

from jacinle.utils.meta import dict_deep_kv
from jacinle.utils.printing import stformat, kvformat

from .fs import as_file_descriptor, io_function_registry

__all__ = [
    'iter_txt',
    'pretty_dump', 'pretty_load',
    'dumps_json', 'dump_json', 'loads_json', 'load_json',
    'dumps_xml', 'dump_xml', 'loads_xml', 'load_xml',
    'dumps_yaml', 'dump_yaml', 'loads_yaml', 'load_yaml',
    'dumps_txt', 'dump_txt',
    'dumps_struct', 'dump_struct',
    'dumps_kv', 'dump_kv',
    'dumps_env', 'dump_env'
]


def iter_txt(fd, strip=True):
    """
    Iterate lines from fp from file.

    Args:
        fd: (todo): write your description
        strip: (bool): write your description
    """
    for line in as_file_descriptor(fd):
        line_strip = line.strip()
        if line_strip == '':
            continue
        yield line_strip if strip else line


def loads_json(value):
    """
    Deserialize a json string.

    Args:
        value: (str): write your description
    """
    return json.loads(value)


def loads_xml(value, **kwargs):
    """
    Deserialize an xml string into python object.

    Args:
        value: (str): write your description
    """
    return _xml2dict(et.fromstring(value), **kwargs)


def loads_yaml(value):
    """
    Parse a yaml document.

    Args:
        value: (todo): write your description
    """
    return yaml.load(value)


def dumps_txt(value):
    """
    Serialize a string as a text string.

    Args:
        value: (todo): write your description
    """
    assert isinstance(value, collections.Iterable), 'dump(s) txt supports only list as input.'
    with _io.StringIO() as buf:
        for v in value:
            v = str(v)
            buf.write(v)
            if v[-1] != '\n':
                buf.write('\n')
        return buf.getvalue()


def dumps_json(value, compressed=True):
    """
    Serializes the object as a json.

    Args:
        value: (str): write your description
        compressed: (bool): write your description
    """
    if compressed:
        return json.dumps(value, cls=JsonObjectEncoder)
    return json.dumps(value, cls=JsonObjectEncoder, sort_keys=True, indent=4, separators=(',', ': '))


def pretty_dumps_json(value, compressed=False):
    """
    Pretty print a human readable to a string.

    Args:
        value: (todo): write your description
        compressed: (bool): write your description
    """
    return dumps_json(value, compressed=compressed)


def dumps_xml(value, **kwargs):
    """
    Serialize an xml string to a string.

    Args:
        value: (todo): write your description
    """
    return _dict2xml(value, **kwargs)


def dumps_yaml(value):
    """
    Serialize value ascii formatted string.

    Args:
        value: (todo): write your description
    """
    return yaml.dump(value, width=80, indent=4)


def dumps_struct(value):
    """
    Convert a struct to a string.

    Args:
        value: (todo): write your description
    """
    return stformat(value)


def dumps_kv(value):
    """
    Convert a kvformat to a string.

    Args:
        value: (todo): write your description
    """
    return kvformat(value)


def dumps_env(value):
    """
    Convert a value as a string.

    Args:
        value: (todo): write your description
    """
    return '\n'.join(['{} = {}'.format(k, v) for k, v in dict_deep_kv(value)])


def _wrap_load(loads_func):
    """
    Decorator for load a function.

    Args:
        loads_func: (todo): write your description
    """
    @functools.wraps(loads_func)
    def load(file, **kwargs):
        """
        Deserialize file descriptor from file - like object.

        Args:
            file: (str): write your description
        """
        with as_file_descriptor(file, 'r') as f:
            return loads_func(f.read(), **kwargs)

    load.__name__ = loads_func.__name__[:-1]
    load.__qualname__ = loads_func.__qualname__[:-1]
    return load


def _wrap_dump(dumps_func):
    """
    Decorator to dump method.

    Args:
        dumps_func: (todo): write your description
    """
    @functools.wraps(dumps_func)
    def dump(file, obj, **kwargs):
        """
        Serialize obj to file - like object.

        Args:
            file: (str): write your description
            obj: (dict): write your description
        """
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

pretty_dump_json = _wrap_dump(pretty_dumps_json)


for registry in ['load', 'pretty_load']:
    io_function_registry.register(registry, '.json', load_json)
    io_function_registry.register(registry, '.xml',  load_xml)
    io_function_registry.register(registry, '.yaml', load_yaml)
    io_function_registry.register(registry, '.yml',  load_yaml)


for registry in ['dump', 'pretty_dump']:
    io_function_registry.register(registry, '.txt',    dump_txt)
    if registry == 'pretty_dump':
        io_function_registry.register(registry, '.json',   pretty_dump_json)
    else:
        io_function_registry.register(registry, '.json',   dump_json)
    io_function_registry.register(registry, '.xml',    dump_xml)
    io_function_registry.register(registry, '.yaml',   dump_yaml)
    io_function_registry.register(registry, '.yml',    dump_yaml)
    io_function_registry.register(registry, '.struct', dump_struct)
    io_function_registry.register(registry, '.kv',     dump_kv)
    io_function_registry.register(registry, '.env',    dump_env)


def pretty_load(file, **kwargs):
    """
    Decorator to load a file from a file - like object.

    Args:
        file: (str): write your description
    """
    return io_function_registry.dispatch('pretty_load', file, **kwargs)


def pretty_dump(file, obj, **kwargs):
    """
    Dump obj as a file - like object.

    Args:
        file: (str): write your description
        obj: (todo): write your description
    """
    return io_function_registry.dispatch('pretty_dump', file, obj, **kwargs)


def _dict2xml(d, indent=4, *, root_node=None, root_indent=0, name_key='__name__', attribute_key='__attribute__'):
    """Adapted from: https://gist.github.com/reimund/5435343/"""

    indent_str = '\n' + ' ' * (indent * root_indent)

    if root_node is None and name_key is not None:
        root_node = d[name_key]

    wrap = False if root_node is None or isinstance(d, list) else True
    root = 'data' if root_node is None else root_node
    root_singular = root[:-1] if 's' == root[-1] and root_node is None else root
    xml = ''
    children = []

    if isinstance(d, dict):
        for key, value in d.items():
            if attribute_key is not None and key == attribute_key:
                continue
            if name_key is not None and key == name_key:
                continue

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

    if attribute_key is not None and attribute_key in d:
        for key, value in d[attribute_key].items():
            xml += f' {key}="{value}"'

    if wrap or isinstance(d, dict):
        xml = indent_str + '<' + root + xml + end_tag

    if len(children) > 0:
        for child in children:
            xml = xml + child

        if wrap or isinstance(d, dict):
            xml = xml + indent_str + '</' + root + '>'

    return xml


def _xml2dict(element, name_key='__name__', attribute_key='__attribute__'):
    """
    Convert an xml element into dict.

    Args:
        element: (todo): write your description
        name_key: (str): write your description
        attribute_key: (str): write your description
    """
    output_dict = {}

    if name_key is not None:
        output_dict[name_key] = element.tag

    if attribute_key is None:
        output_dict.update(element.attrib)
    else:
        output_dict[attribute_key] = element.attrib

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


class JsonObjectEncoder(json.JSONEncoder):
    """Adapted from https://stackoverflow.com/a/35483750"""

    def default(self, obj):
        """
        Default json - serialized object.

        Args:
            self: (todo): write your description
            obj: (todo): write your description
        """
        if hasattr(obj, '__jsonify__'):
            json_object = obj.__jsonify__()
            if isinstance(json_object, six.string_types):
                return json_object
            return self.encode(json_object)
        else:
            raise TypeError("Object of type '%s' is not JSON serializable." % obj.__class__.__name__)

        if hasattr(obj, '__dict__'):
            d = dict(
                (key, value)
                for key, value in inspect.getmembers(obj)
                if not key.startswith("__")
                and not inspect.isabstract(value)
                and not inspect.isbuiltin(value)
                and not inspect.isfunction(value)
                and not inspect.isgenerator(value)
                and not inspect.isgeneratorfunction(value)
                and not inspect.ismethod(value)
                and not inspect.ismethoddescriptor(value)
                and not inspect.isroutine(value)
            )
            return self.default(d)

        return obj

