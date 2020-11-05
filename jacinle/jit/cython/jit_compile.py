#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : jit_compile.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/06/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import uuid
import tempfile
import re
import ast
import inspect
import sys
import importlib
import subprocess

import jacinle.io as io
from jacinle.utils.naming import func_name

__all__ = ['CythonCompiledFunction', 'CythonJITCompiler', 'jit_cython']


class CythonCompiledFunction(object):
    def __init__(self, name, func_name, build_dir, py_source, extra_args):
        """
        Initialize the pyx function.

        Args:
            self: (todo): write your description
            name: (str): write your description
            func_name: (str): write your description
            build_dir: (str): write your description
            py_source: (str): write your description
            extra_args: (str): write your description
        """
        self.name = name
        self.func_name = func_name
        self.build_dir = build_dir
        self.py_source = py_source
        self.extra_args = extra_args

        self.pyx_source = None
        self.pyx_source_cdef = None
        self.dependencies = None
        self.func = None

    def __call__(self, *args, **kwargs):
        """
        Calls a function call.

        Args:
            self: (todo): write your description
        """
        return self.func(*args, **kwargs)

    def __repr__(self):
        """
        Return a human - readable string.

        Args:
            self: (todo): write your description
        """
        return '<CythonCompiledFunction {} at 0x{:x}>'.format(self.func_name, id(self))


class _DependencyNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        """
        Initialize the dependency.

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.dependencies = list()

    def visit_Call(self, node):
        """
        Assign a function call.

        Args:
            self: (todo): write your description
            node: (todo): write your description
        """
        if isinstance(node.func, ast.Name):
            self.dependencies.append(node.func.id)
        else:
            assert isinstance(node.func, ast.Attribute), type(node.func)
            self.dependencies.append(node.func.value.id)


    def __call__(self, node):
        """
        Recursively.

        Args:
            self: (todo): write your description
            node: (todo): write your description
        """
        self.dependencies = list()
        self.generic_visit(node)
        return self.dependencies


class CythonJITCompiler(object):
    source_header  = """
import numpy as np
cimport numpy as np
cimport cython
"""
    source_footer = ''
    setup_template = """
from setuptools import setup, Extension
import numpy as np

setup(
    name='{func_name}',
    ext_modules=[
        Extension('{func_name}', sources=['{func_name}.pyx'], include_dirs=[np.get_include()], annotate=True)
    ]
)
"""

    all_funcs = dict()

    def __init__(self):
        """
        Initialize the object

        Args:
            self: (todo): write your description
        """
        pass

    def compile(self, f=None, *, name=None, force_update=False, boundscheck=True, wraparound=True):
        """
        Compile cython function.

        Args:
            self: (todo): write your description
            f: (str): write your description
            name: (str): write your description
            force_update: (bool): write your description
            boundscheck: (bool): write your description
            wraparound: (str): write your description
        """
        def wrapper(func):
            """
            Decorator for the cython.

            Args:
                func: (callable): write your description
            """
            name = self.get_name(func)
            func_name = func.__name__
            build_dir = self.get_build_dir(func)
            py_source = self.get_source(func)
            func = CythonCompiledFunction(
                name, func_name, build_dir, py_source,
                extra_args=dict(boundscheck=boundscheck, wraparound=wraparound)
            )

            if self.check_source_updated(build_dir, py_source, force_update=force_update):
                self.gen_pyx_source(func)
                cython_source = self.gen_cython_source(func)
                self.write_cython(name, func_name, build_dir, cython_source)
                self.build_func(func_name, build_dir)
            else:
                pass

            func.func = self.load_func(func_name, build_dir)
            self.all_funcs[name] = func
            return func

        if f is not None:
            return wrapper(f)
        return wrapper

    def get_name(self, func):
        """
        Return the name of a function.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        name = func_name(func)
        return name

    def get_source(self, func):
        """
        Return the source of the given function.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        source = inspect.getsource(func)
        return source

    def gen_pyx_source(self, func):
        """
        Generate a pyx source.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        if func.pyx_source is not None:
            return

        func.pyx_source = func.py_source
        self.cleanup_source(func)
        self.resolve_dependencies(func)
        self.optimize_cdef(func)
        self.add_extra_args(func)

    def gen_pyx_source_cdef(self, func):
        """
        Generate pyx_source_source object.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        self.gen_pyx_source(func)
        self.optimize_cdef_function(func)

    def gen_cython_source(self, func):
        """
        Generate c ++ function that returns the source file.

        Args:
            self: (todo): write your description
            func: (callable): write your description
        """
        visited = set()
        cython_source = ''

        def visit(f):
            """
            Add a new function to pyx.

            Args:
                f: (todo): write your description
            """
            visited.add(f.name)

            self.gen_pyx_source_cdef(f)
            for dep in f.dependencies:
                dep_name = f.name[:-len(f.func_name)] + dep
                if dep_name in self.all_funcs and dep_name not in visited:
                    visit(self.all_funcs[dep_name])

            nonlocal cython_source
            if f == func:
                cython_source += f.pyx_source
            else:
                cython_source += f.pyx_source_cdef

        try:
            visit(func)
        finally:
            del visit

        cython_source = self.source_header + '\n' + cython_source + '\n' + self.source_footer
        return cython_source

    decorator_cleanup_re = re.compile('^([ \t]*)@jit_cython.*\n', flags=re.M)

    def cleanup_source(self, func):
        """
        Cleanup a function.

        Args:
            self: (todo): write your description
            func: (callable): write your description
        """
        func.pyx_source = self.decorator_cleanup_re.sub('', func.pyx_source)

    dependency_visitor = _DependencyNodeVisitor()

    def resolve_dependencies(self, func):
        """
        Resolve the given function.

        Args:
            self: (todo): write your description
            func: (callable): write your description
        """
        tree = ast.parse(func.pyx_source)
        assert isinstance(tree.body[0], ast.FunctionDef)
        func.dependencies = self.dependency_visitor(tree.body[0])

    # TODO(Jiayuan Mao @ 04/06): exclude args.
    cdef_re = re.compile('^([ \t]+)([a-zA-Z_][a-zA-Z_0-9]*)( )*:( )*\'?(([a-zA-Z_][a-zA-Z_0-9]*\.)*([a-zA-Z_][a-zA-Z_0-9]*))(\[.*?\])?\'?', flags=re.M)
    args_re = re.compile('([a-zA-Z_][a-zA-Z_0-9]*)( )*:( )*\'?(([a-zA-Z_][a-zA-Z_0-9]*\.)*([a-zA-Z_][a-zA-Z_0-9]*))(\[.*?\])?\'?', flags=re.M)

    def optimize_cdef(self, func):
        """
        Optimize the function.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        func.pyx_source = self.cdef_re.sub('\\1cdef \\5\\8 \\2', func.pyx_source)
        func.pyx_source = self.args_re.sub('\\4\\7 \\1', func.pyx_source)

    def add_extra_args(self, func):
        """
        Add extra arguments.

        Args:
            self: (todo): write your description
            func: (callable): write your description
        """
        func.pyx_source = '@cython.boundscheck({})\n'.format(func.extra_args['boundscheck']) + \
                '@cython.wraparound({})\n'.format(func.extra_args['wraparound']) + func.pyx_source

    cdef_function_re = re.compile('^([ \t]*)def (.*?)( )*->( )*\'?(([a-zA-Z_][a-zA-Z_0-9]*\.)*([a-zA-Z_][a-zA-Z_0-9]*))\'?(\[.*?\])?', flags=re.M)
    def optimize_cdef_function(self, func):
        """
        Optimize the function.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        func.pyx_source_cdef = self.cdef_function_re.sub('\\1cdef \\5\\8 \\2', func.pyx_source)

    def check_source_updated(self, build_dir, source, force_update=False):
        """
        Checks if_source_dir is updated.

        Args:
            self: (todo): write your description
            build_dir: (str): write your description
            source: (str): write your description
            force_update: (bool): write your description
        """
        source_filename = osp.join(build_dir, 'source.py')
        if osp.exists(source_filename) and not force_update:
            with open(source_filename) as f:
                old_source = f.read()
            if source == old_source:
                return False
        with open(source_filename, 'w') as f:
            f.write(source)
        return True

    def check_module_exist(self, func_name, build_dir):
        """
        Check if a build directory exists.

        Args:
            self: (todo): write your description
            func_name: (str): write your description
            build_dir: (str): write your description
        """
        for x in os.listdir(build_dir):
            # TODO(Jiayuan Mao @ 04/06): support Windows.
            if x.startswith(func_name) and x.endswith('.so'):
                return True
        return False

    def write_cython(self, name, func_name, build_dir, source):
        """
        Write c ++ source file

        Args:
            self: (todo): write your description
            name: (str): write your description
            func_name: (str): write your description
            build_dir: (str): write your description
            source: (str): write your description
        """
        cython_filename = osp.join(build_dir, func_name + '.pyx')
        with open(cython_filename, 'w') as f:
            f.write(source)
        setup_filename = osp.join(build_dir, 'setup.py')
        with open(setup_filename, 'w') as f:
            f.write(self.setup_template.format(func_name=func_name))

    def build_func(self, func_name, build_dir):
        """
        Build the build function.

        Args:
            self: (todo): write your description
            func_name: (str): write your description
            build_dir: (str): write your description
        """
        subprocess.check_call(['python','setup.py', 'build_ext', '--inplace'], cwd=build_dir)

    def load_func(self, func_name, build_dir):
        """
        Loads a python module.

        Args:
            self: (todo): write your description
            func_name: (str): write your description
            build_dir: (str): write your description
        """
        if not self.check_module_exist(func_name, build_dir):
            self.build_func(func_name, build_dir)

        sys.path.insert(0, build_dir)
        module = importlib.import_module(func_name)
        sys.path = sys.path[1:]
        return getattr(module, func_name)

    def get_build_dir(self, func):
        """
        Return the build directory.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        name = self.get_name(func)
        build_dir = osp.join(tempfile.gettempdir(), 'jacinle_cython', name)
        io.mkdir(build_dir)
        return build_dir


_compiler = CythonJITCompiler()


def jit_cython(f=None, *, name=None, force_update=False, boundscheck=True, wraparound=True):
    """
    Compile a compiler

    Args:
        f: (todo): write your description
        name: (str): write your description
        force_update: (bool): write your description
        boundscheck: (bool): write your description
        wraparound: (todo): write your description
    """
    return _compiler.compile(f, name=name, force_update=force_update, boundscheck=boundscheck, wraparound=wraparound)

