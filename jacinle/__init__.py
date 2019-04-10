#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


from jacinle.utils.init import init_main

init_main()

del init_main

from jacinle.cli.argument import JacArgumentParser
from jacinle.cli.keyboard import yes_or_no, maybe_mkdir
from jacinle.concurrency.pool import TQDMPool
from jacinle.config.environ import load_env, has_env, get_env, set_env, with_env
from jacinle.logging import get_logger
from jacinle.utils.cache import cached_property, cached_result, fs_cached_result
from jacinle.utils.container import G, g, GView, SlotAttrObject, OrderedSet
from jacinle.utils.context import EmptyContext
from jacinle.utils.defaults import defaults_manager, wrap_custom_as_default, gen_get_default, gen_set_default
from jacinle.utils.deprecated import deprecated
from jacinle.utils.enum import JacEnum
from jacinle.utils.exception import format_exc
from jacinle.utils.env import jac_getenv, jac_is_verbose, jac_is_debug
from jacinle.utils.imp import load_module, load_module_filename, load_source
from jacinle.utils.meta import (
        gofor,
        run_once, try_run,
        map_exec, filter_exec, stmap,
        method2func, map_exec_method,
        decorator_with_optional_args,
        cond_with,
        merge_iterable,
        dict_deep_update, dict_deep_kv, dict_deep_keys,
        assert_instance, assert_none, assert_notnone,
        notnone_property, synchronized
)
from jacinle.utils.meter import GroupMeters
from jacinle.utils.naming import class_name, func_name, method_name, class_name_of_method
from jacinle.utils.network import get_local_addr
from jacinle.utils.numeric import safe_sum, mean, std, rms, prod, divup
from jacinle.utils.printing import stprint, stformat, kvprint, kvformat, print_to_string
from jacinle.utils.tqdm import get_current_tqdm, tqdm, tqdm_pbar, tqdm_gofor, tqdm_zip

# For debug purpose.
def embed(*args, **kwargs):
    from IPython import embed
    embed(*args, **kwargs)

def pprint(*args, **kwargs):
    from pprint import pprint
    pprint(*args, **kwargs)

from sys import exit

