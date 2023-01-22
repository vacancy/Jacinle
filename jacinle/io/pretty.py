#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pretty.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/15/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Functions to dump Python objects into human-readable formats."""

import io as _io
import json
import functools
import collections
import xml.etree.ElementTree as et
import yaml
import six
import inspect

from typing import Optional, Any, Iterable, Sequence, List, Dict
from jacinle.utils.meta import dict_deep_kv
from jacinle.utils.printing import stformat, kvformat

from .fs import as_file_descriptor, io_function_registry

__all__ = [
    'iter_txt',
    'pretty_dump', 'pretty_load',
    'dumps_json', 'dump_json', 'loads_json', 'load_json', 'pretty_dumps_json', 'pretty_dump_json',
    'dumps_jsonc', 'dump_jsonc', 'loads_jsonc', 'load_jsonc',
    'dumps_xml', 'dump_xml', 'loads_xml', 'load_xml',
    'dumps_yaml', 'dump_yaml', 'loads_yaml', 'load_yaml',
    'dumps_txt', 'dump_txt',
    'dumps_struct', 'dump_struct',
    'dumps_kv', 'dump_kv',
    'dumps_env', 'dump_env'
]


def iter_txt(fd: _io.IOBase, strip: bool = True) -> Iterable[str]:
    """Iterate over lines in a text file. This function will ignore empty lines.

    Args:
        fd: a file descriptor.
        strip: whether to strip the line.
    """
    for line in as_file_descriptor(fd):
        line_strip = line.strip()
        if line_strip == '':
            continue
        yield line_strip if strip else line


def loads_json(value: str) -> Any:
    """Load a JSON object from a string."""
    return json.loads(value)


def loads_jsonc(value: str) -> List[Dict]:
    """Load a list of JSON dictionaries from a string. This function supports multiple JSON objects in a single string, separated by newlines.
    Note that this function only support a list of plain dictionaries. Do not use this function to load JSON objects with
    nested lists or dictionaries.

    Args:
        value: a string.

    Returns:
        a list of dictionaries.
    """
    strings = value.split('}\n{')
    ret = []
    for i, s in enumerate(strings):
        if i > 0:
            s = '{' + s
        if i < len(strings) - 1:
            s += '}'
        ret.append(json.loads(s))
    return ret


def loads_xml(value: str, name_key: str = '__name__', attribute_key: Optional[str] = '__attribute__') -> Dict:
    """Load an XML object from a string. It will return a dictionary as the root node.
    For each node, it will have a key named "__name__" as the tag name, and a key "__attributes__" as a dictionary of attributes.
    Each child node will be a nested dictionary under the root node. If there are multiple child nodes with the same tag name,
    they will be stored in a list.

    Args:
        value: a string.
        name_key: the key name for the tag name.
        attribute_key: the key name for the attributes. If set to None, the attributes will be stored in the root node.

    Returns:
        a dictionary.
    """
    return _xml2dict(et.fromstring(value), name_key=name_key, attribute_key=attribute_key)


def loads_yaml(value: str) -> Any:
    """Load a YAML object from a string."""
    return yaml.safe_load(value)


def dumps_txt(value: Iterable[str]) -> str:
    """Dump a list of strings into a string, separated by newlines.

    Args:
        value: a list of strings.

    Returns:
        a string.
    """
    assert isinstance(value, collections.Iterable), 'dump(s) txt supports only list as input.'
    with _io.StringIO() as buf:
        for v in value:
            v = str(v)
            buf.write(v)
            if v[-1] != '\n':
                buf.write('\n')
        return buf.getvalue()


def dumps_json(value: Any, compressed: bool = True) -> str:
    """Dump a JSON object into a string. In addition to the standard JSON format, this function also supports

    - ``__jsonify__``: an instance method for objects that returns a JSON-serializable object.
    - For classes, it will store ``__dict__`` as the JSON object.

    Note that both features can not be preserved when loading the JSON object back.

    Args:
        value: the object to dump.
        compressed: whether to use compressed format. If set to False, the dumped string will be pretty-printed.

    Returns:
        a string.
    """
    if compressed:
        return json.dumps(value, cls=_JsonObjectEncoder)
    return json.dumps(value, cls=_JsonObjectEncoder, sort_keys=True, indent=4, separators=(',', ': '))


def pretty_dumps_json(value: Any, compressed: bool = False) -> str:
    """Dump a JSON object into a string, with pretty-printing."""
    return dumps_json(value, compressed=compressed)


def dumps_jsonc(value: Iterable[Dict]) -> str:
    """Dump a list of dictionary into a JSON string, separated by new lines, with compressed format."""
    assert isinstance(value, Sequence)
    ret = ''
    for v in value:
        ret += pretty_dumps_json(v) + '\n'
    return ret


def dumps_xml(value: Dict, **kwargs) -> str:
    """Dump an XML object into a string."""
    return _dict2xml(value, **kwargs)


def dumps_yaml(value: Any) -> str:
    """Dump a YAML object into a string."""
    return yaml.dump(value, width=80, indent=4)


def dumps_struct(value):
    """Dump a structured object into a string, using :meth:`jacinle.utils.printing.stformat`."""
    return stformat(value)


def dumps_kv(value):
    """Dump a structured object into a string, using :meth:`jacinle.utils.printing.kvformat`."""
    return kvformat(value)


def dumps_env(value):
    """Dump a structured object into a string, similar to :meth:`os.environ`."""
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
    dump.__doc__ = dumps_func.__doc__.replace('a string', 'a file')
    return dump


load_json = _wrap_load(loads_json)
load_jsonc = _wrap_load(loads_jsonc)
load_xml = _wrap_load(loads_xml)
load_yaml = _wrap_load(loads_yaml)

dump_txt = _wrap_dump(dumps_txt)
dump_json = _wrap_dump(dumps_json)
dump_jsonc = _wrap_dump(dumps_jsonc)
dump_xml = _wrap_dump(dumps_xml)
dump_yaml = _wrap_dump(dumps_yaml)
dump_struct = _wrap_dump(dumps_struct)
dump_kv = _wrap_dump(dumps_kv)
dump_env = _wrap_dump(dumps_env)

pretty_dump_json = _wrap_dump(pretty_dumps_json)


for registry in ['load', 'pretty_load']:
    io_function_registry.register(registry, '.json', load_json)
    io_function_registry.register(registry, '.jsonc', load_jsonc)
    io_function_registry.register(registry, '.xml',  load_xml)
    io_function_registry.register(registry, '.yaml', load_yaml)
    io_function_registry.register(registry, '.yml',  load_yaml)


for registry in ['dump', 'pretty_dump']:
    io_function_registry.register(registry, '.txt',    dump_txt)
    if registry == 'pretty_dump':
        io_function_registry.register(registry, '.json',   pretty_dump_json)
    else:
        io_function_registry.register(registry, '.json',   dump_json)
    io_function_registry.register(registry, '.jsonc',   dump_jsonc)
    io_function_registry.register(registry, '.xml',    dump_xml)
    io_function_registry.register(registry, '.yaml',   dump_yaml)
    io_function_registry.register(registry, '.yml',    dump_yaml)
    io_function_registry.register(registry, '.struct', dump_struct)
    io_function_registry.register(registry, '.kv',     dump_kv)
    io_function_registry.register(registry, '.env',    dump_env)


def pretty_load(file, **kwargs):
    """Load a file with pretty-printing."""
    return io_function_registry.dispatch('pretty_load', file, **kwargs)


def pretty_dump(file, obj, **kwargs):
    """Dump a file with pretty-printing."""
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


class _JsonObjectEncoder(json.JSONEncoder):
    """Adapted from https://stackoverflow.com/a/35483750"""

    def default(self, obj):
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

