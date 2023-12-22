#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""The Jacinle library.

This main library contains a set of useful utility functions and classes for general Python scripting.

There are a few automatically imported submodules that can be accessed by ``jacinle.<submodule>``.

.. rubric:: Command Line Tools

.. autosummary::

    ~jacinle.cli.argument.JacArgumentParser
    ~jacinle.cli.keyboard.yes_or_no
    ~jacinle.cli.keyboard.maybe_mkdir
    ~jacinle.cli.git.git_guard

.. rubric:: Logging

.. autosummary::

    ~jacinle.logging.get_logger
    ~jacinle.logging.set_logger_output_file

.. rubric:: Configuration

See :doc:`jacinle.config.environ_v2` for more details.

.. autosummary::

    ~jacinle.config.environ_v2.configs
    ~jacinle.config.environ_v2.def_configs
    ~jacinle.config.environ_v2.def_configs_func
    ~jacinle.config.environ_v2.set_configs
    ~jacinle.config.environ_v2.set_configs_func
    ~jacinle.utils.env.jac_getenv
    ~jacinle.utils.env.jac_is_verbose
    ~jacinle.utils.env.jac_is_debug

.. rubric:: Utilities (Core)

.. autosummary::

    ~jacinle.utils.context.EmptyContext
    ~jacinle.utils.context.KeyboardInterruptContext
    ~jacinle.utils.enum.JacEnum
    ~jacinle.utils.meta.Clock

    ~jacinle.utils.deprecated.deprecated

    ~jacinle.utils.imp.load_module
    ~jacinle.utils.imp.load_module_filename
    ~jacinle.utils.imp.load_source

    ~jacinle.utils.meta.gofor
    ~jacinle.utils.meta.run_once
    ~jacinle.utils.meta.try_run
    ~jacinle.utils.meta.map_exec
    ~jacinle.utils.meta.filter_exec
    ~jacinle.utils.meta.first
    ~jacinle.utils.meta.first_n
    ~jacinle.utils.meta.stmap
    ~jacinle.utils.meta.method2func
    ~jacinle.utils.meta.map_exec_method
    ~jacinle.utils.meta.decorator_with_optional_args
    ~jacinle.utils.meta.cond_with
    ~jacinle.utils.meta.cond_with_group
    ~jacinle.utils.meta.merge_iterable
    ~jacinle.utils.meta.dict_deep_update
    ~jacinle.utils.meta.dict_deep_kv
    ~jacinle.utils.meta.dict_deep_keys
    ~jacinle.utils.meta.assert_instance
    ~jacinle.utils.meta.assert_none
    ~jacinle.utils.meta.assert_notnone
    ~jacinle.utils.meta.notnone_property
    ~jacinle.utils.meta.synchronized
    ~jacinle.utils.meta.timeout
    ~jacinle.utils.meta.make_dummy_func
    ~jacinle.utils.meta.repr_from_str

    ~jacinle.utils.inspect.class_name
    ~jacinle.utils.inspect.func_name
    ~jacinle.utils.inspect.method_name
    ~jacinle.utils.inspect.class_name_of_method

    ~jacinle.utils.printing.indent_text
    ~jacinle.utils.printing.stprint
    ~jacinle.utils.printing.stformat
    ~jacinle.utils.printing.kvprint
    ~jacinle.utils.printing.kvformat
    ~jacinle.utils.printing.print_to_string
    ~jacinle.utils.printing.print_to
    ~jacinle.utils.printing.suppress_stdout
    ~jacinle.utils.printing.suppress_stderr
    ~jacinle.utils.printing.suppress_output

.. rubric:: Utilities (IO)

.. autosummary::

    ~jacinle.io.fs.load
    ~jacinle.io.fs.dump
    ~jacinle.io.fs.load_json
    ~jacinle.io.fs.dump_json
    ~jacinle.io.fs.load_pkl
    ~jacinle.io.fs.dump_pkl
    ~jacinle.io.fs.lsdir
    ~jacinle.io.fs.mkdir

.. rubric:: Utilities (Cache)

.. autosummary::

    ~jacinle.utils.cache.cached_property
    ~jacinle.utils.cache.cached_result
    ~jacinle.utils.cache.fs_cached_result

.. rubric:: Utilities (TQDM)

.. autosummary::

    ~jacinle.utils.tqdm.get_current_tqdm
    ~jacinle.utils.tqdm.tqdm
    ~jacinle.utils.tqdm.tqdm_pbar
    ~jacinle.utils.tqdm.tqdm_gofor
    ~jacinle.utils.tqdm.tqdm_zip
    ~jacinle.concurrency.pool.TQDMPool

.. rubric:: Utilities (Math)

.. autosummary::

    ~jacinle.utils.meter.GroupMeters
    ~jacinle.utils.numeric.safe_sum
    ~jacinle.utils.numeric.mean
    ~jacinle.utils.numeric.std
    ~jacinle.utils.numeric.rms
    ~jacinle.utils.numeric.prod
    ~jacinle.utils.numeric.divup
    ~jacinle.random.rng.reset_global_seed
    ~jacinle.random.rng.seed
    ~jacinle.random.rng.with_seed

.. rubric:: Utilities (Container)

.. autosummary::

    ~jacinle.utils.container.g
    ~jacinle.utils.container.G
    ~jacinle.utils.container.GView
    ~jacinle.utils.container.SlotAttrObject
    ~jacinle.utils.container.OrderedSet

.. rubric:: Utilities (Defaults)

See :doc:`jacinle.utils.defaults` for more details.

.. autosummary::

    ~jacinle.utils.defaults.defaults_manager
    ~jacinle.utils.defaults.wrap_custom_as_default
    ~jacinle.utils.defaults.gen_get_default
    ~jacinle.utils.defaults.gen_set_default
    ~jacinle.utils.defaults.option_context
    ~jacinle.utils.defaults.FileOptions
    ~jacinle.utils.defaults.default_args
    ~jacinle.utils.defaults.ARGDEF

.. rubric:: Utilities (Exception and Debugging)

.. autosummary::

    ~jacinle.utils.debug.hook_exception_ipdb
    ~jacinle.utils.debug.exception_hook
    ~jacinle.utils.debug.timeout_ipdb
    ~jacinle.utils.debug.log_function
    ~jacinle.utils.debug.profile
    ~jacinle.utils.debug.time
    ~jacinle.utils.exception.format_exc

.. rubric:: Utilities (Network and Misc)

.. autosummary::

    ~jacinle.utils.network.get_local_addr
    ~jacinle.utils.uid.gen_time_string
    ~jacinle.utils.uid.gen_uuid4
"""

from jacinle.utils.init import init_main

init_main()

del init_main

from jacinle.utils.env import jac_getenv, jac_is_verbose, jac_is_debug

if jac_getenv('IMPORT_ALL', 'true', 'bool'):
    from jacinle.cli.argument import JacArgumentParser
    from jacinle.cli.keyboard import yes_or_no, maybe_mkdir
    from jacinle.cli.git import git_guard
    from jacinle.concurrency.pool import TQDMPool
    from jacinle.config.environ_v2 import configs, def_configs, def_configs_func, set_configs, set_configs_func
    from jacinle.logging import get_logger, set_logger_output_file
    from jacinle.utils.cache import cached_property, cached_result, fs_cached_result
    from jacinle.utils.container import G, g, GView, SlotAttrObject, OrderedSet
    from jacinle.utils.context import EmptyContext, KeyboardInterruptContext
    from jacinle.utils.debug import hook_exception_ipdb, exception_hook, timeout_ipdb, log_function, profile, time
    from jacinle.utils.defaults import (
        defaults_manager, wrap_custom_as_default, gen_get_default, gen_set_default,
        option_context, FileOptions,
        default_args, ARGDEF
    )
    from jacinle.utils.deprecated import deprecated
    from jacinle.utils.enum import JacEnum
    from jacinle.utils.env import jac_getenv, jac_is_debug, jac_is_verbose
    from jacinle.utils.exception import format_exc
    from jacinle.utils.imp import load_module, load_module_filename, load_source
    from jacinle.utils.meta import (
        gofor,
        run_once, try_run,
        map_exec, filter_exec, first, first_n, stmap,
        method2func, map_exec_method,
        decorator_with_optional_args,
        cond_with, cond_with_group,
        merge_iterable,
        dict_deep_update, dict_deep_kv, dict_deep_keys,
        assert_instance, assert_none, assert_notnone,
        notnone_property, synchronized, timeout, Clock, make_dummy_func,
        repr_from_str
    )
    from jacinle.utils.meter import AverageMeter, GroupMeters
    from jacinle.utils.inspect import class_name, func_name, method_name, class_name_of_method
    from jacinle.utils.network import get_local_addr
    from jacinle.utils.numeric import safe_sum, mean, std, rms, prod, divup
    from jacinle.utils.printing import indent_text, stprint, stformat, kvprint, kvformat, print_to_string, print_to, suppress_stdout, suppress_stderr, suppress_output
    from jacinle.utils.tqdm import get_current_tqdm, tqdm, tqdm_pbar, tqdm_gofor, tqdm_zip
    from jacinle.utils.uid import gen_time_string, gen_uuid4

    from jacinle.io.fs import load, dump, mkdir, lsdir
    from jacinle.io.fs import load_pkl, dump_pkl
    from jacinle.io.pretty import load_json, dump_json
    from jacinle.random import reset_global_seed, seed, with_seed

    import jacinle.cli.git as git
    import jacinle.io as io
    import jacinle.nd as nd
    import jacinle.random as random

    try:
        from IPython import embed
    except ImportError:
        pass

    try:
        from pprint import pprint
    except ImportError:
        pass

    try:
        from tabulate import tabulate
    except ImportError:
        pass

    JAC_VERBOSE = jac_is_verbose()
    JAC_DEBUG = jac_is_debug()

