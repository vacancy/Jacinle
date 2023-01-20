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

.. rubric:: Core Submodules

.. autosummary::

    jacinle.git
    jacinle.io
    jacinle.nd
    jacinle.random

.. rubric:: Command Line Tools

.. autosummary::

    JacArgumentParser
    yes_or_no
    maybe_mkdir
    git_guard

.. rubric:: Logging

.. autosummary::

    get_logger
    set_logger_output_file

.. rubric:: Configuration

.. autosummary::

    configs
    def_configs
    def_configs_func
    set_configs
    set_configs_func
    jac_getenv
    jac_is_verbose
    jac_is_debug

.. rubric:: Utilities (Core)

.. autosummary::

    EmptyContext
    KeyboardInterruptContext
    JacEnum
    Clock

    deprecated

    load_module
    load_module_filename
    load_source

    gofor
    run_once
    try_run
    map_exec
    filter_exec
    first
    first_n
    stmap
    method2func
    map_exec_method
    decorator_with_optional_args
    cond_with
    cond_with_group
    merge_iterable
    dict_deep_update
    dict_deep_kv
    dict_deep_keys
    assert_instance
    assert_none
    assert_notnone
    notnone_property
    synchronized
    timeout
    make_dummy_func
    repr_from_str

    class_name
    func_name
    method_name
    class_name_of_method

    indent_text
    stprint
    stformat
    kvprint
    kvformat
    print_to_string
    print_to
    suppress_stdout

.. rubric:: Utilities (IO)

.. autosummary::

    load
    dump

.. rubric:: Utilities (Cache)

.. autosummary::

    cached_property
    cached_result
    fs_cached_result

.. rubric:: Utilities (TQDM)

.. autosummary::

    get_current_tqdm
    tqdm
    tqdm_pbar
    tqdm_gofor
    tqdm_zip
    TQDMPool

.. rubric:: Utilities (Math)

.. autosummary::

    GroupMeters
    safe_sum
    mean
    std
    rms
    prod
    divup
    reset_global_seed

.. rubric:: Utilities (Container)

.. autosummary::

    G
    g
    GView
    SlotAttrObject
    OrderedSet

.. rubric:: Utilities (Defaults)

.. autosummary::

    defaults_manager
    wrap_custom_as_default
    gen_get_default
    gen_set_default
    option_context
    FileOptions
    default_args
    ARGDEF

.. rubric:: Utilities (Exception and Debugging)

.. autosummary::

    hook_exception_ipdb
    exception_hook
    timeout_ipdb
    log_function
    profile
    time
    format_exc

.. rubric:: Utilities (Network and Misc)

.. autosummary::

    get_local_addr
    gen_time_string
    gen_uuid4


.. rubric:: Modules
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
    from jacinle.logging.logger import get_logger, set_logger_output_file
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
    from jacinle.utils.naming import class_name, func_name, method_name, class_name_of_method
    from jacinle.utils.network import get_local_addr
    from jacinle.utils.numeric import safe_sum, mean, std, rms, prod, divup
    from jacinle.utils.printing import indent_text, stprint, stformat, kvprint, kvformat, print_to_string, print_to, suppress_stdout
    from jacinle.utils.tqdm import get_current_tqdm, tqdm, tqdm_pbar, tqdm_gofor, tqdm_zip
    from jacinle.utils.uid import gen_time_string, gen_uuid4

    from jacinle.io import load, dump
    from jacinle.random import reset_global_seed

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

    JAC_VERBOSE = jac_is_verbose()
    JAC_DEBUG = jac_is_debug()

