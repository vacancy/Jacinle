#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _inspect_mate.py
# Author : Sanhe Hu
# Email  : maojiayuan@gmail.com
# Date   : 01/03/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Copied from:
    https://gist.github.com/MacHu-GWU/0170849f693aa5f8d129aa03fc358305

``inspect_mate`` provides more methods to get information about class attribute
than the standard library ``inspect``.
This module is Python2/3 compatible, tested under Py2.7, 3.3, 3.4, 3.5, 3.6.
Includes tester function to check:
- is regular attribute
- is property style method
- is regular method, example: ``def method(self, *args, **kwargs)``
- is static method
- is class method
These are 5-kind class attributes.
and getter function to get each kind of class attributes of a class.
"""

__version__ = "0.0.1"
__author__ = "Sanhe Hu"
__license__ = "MIT"


import inspect
import functools


def is_attribute(klass, attr, value=None):
    """Test if a value of a class is attribute. (Not a @property style
    attribute)
    :param klass: the class
    :param attr: attribute name
    :param value: attribute value
    """
    if value is None:
        value = getattr(klass, attr)
    assert getattr(klass, attr) == value

    if not inspect.isroutine(value):
        if not isinstance(value, property):
            return True
    return False


def is_property_method(klass, attr, value=None):
    """Test if a value of a class is @property style attribute.
    :param klass: the class
    :param attr: attribute name
    :param value: attribute value
    """
    if value is None:
        value = getattr(klass, attr)
    assert getattr(klass, attr) == value

    if not inspect.isroutine(value):
        if isinstance(value, property):
            return True
    return False


def is_regular_method(klass, attr, value=None):
    """Test if a value of a class is regular method.
    example::
        class MyClass(object):
            def to_dict(self):
                ...
    :param klass: the class
    :param attr: attribute name
    :param value: attribute value
    """
    if value is None:
        value = getattr(klass, attr)
    assert getattr(klass, attr) == value

    if inspect.isroutine(value):
        if not is_static_method(klass, attr, value) \
                and not is_class_method(klass, attr, value):
            return True

    return False


def is_static_method(klass, attr, value=None):
    """Test if a value of a class is static method.
    example::
        class MyClass(object):
            @staticmethod
            def method():
                ...
    :param klass: the class
    :param attr: attribute name
    :param value: attribute value
    """
    if value is None:
        value = getattr(klass, attr)
    assert getattr(klass, attr) == value

    for cls in inspect.getmro(klass):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                binded_value = cls.__dict__[attr]
                return isinstance(binded_value, staticmethod)
    return False


def is_class_method(klass, attr, value=None):
    """Test if a value of a class is class method.
    example::
        class MyClass(object):
            @classmethod
            def method(cls):
                ...
    :param klass: the class
    :param attr: attribute name
    :param value: attribute value
    """
    if value is None:
        value = getattr(klass, attr)
    assert getattr(klass, attr) == value

    for cls in inspect.getmro(klass):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                binded_value = cls.__dict__[attr]
                if isinstance(binded_value, classmethod):
                    return True
    return False


def _get_members(klass, tester_func, return_builtin):
    """
    :param klass: a class.
    :param tester_func: is_xxx function.
    :param allow_builtin: bool, if False, built-in variable or method such as
      ``__name__``, ``__init__`` will not be returned.
    """
    if not inspect.isclass(klass):
        raise ValueError

    pairs = list()
    for attr, value in inspect.getmembers(klass):
        if tester_func(klass, attr, value):
            if return_builtin:
                pairs.append((attr, value))
            else:
                if not (attr.startswith("__") or attr.endswith("__")):
                    pairs.append((attr, value))

    return pairs


get_attributes = functools.partial(
    _get_members, tester_func=is_attribute, return_builtin=False)
get_attributes.__doc__ = "Get all class attributes members."

get_property_methods = functools.partial(
    _get_members, tester_func=is_property_method, return_builtin=False)
get_property_methods.__doc__ = "Get all property style attributes members."

get_regular_methods = functools.partial(
    _get_members, tester_func=is_regular_method, return_builtin=False)
get_regular_methods.__doc__ = "Get all non static and class method members"

get_static_methods = functools.partial(
    _get_members, tester_func=is_static_method, return_builtin=False)
get_static_methods.__doc__ = "Get all static method attributes members."

get_class_methods = functools.partial(
    _get_members, tester_func=is_class_method, return_builtin=False)
get_class_methods.__doc__ = "Get all class method attributes members."


def get_all_attributes(klass):
    """Get all attribute members (attribute, property style method).
    """
    if not inspect.isclass(klass):
        raise ValueError

    pairs = list()
    for attr, value in inspect.getmembers(
            klass, lambda x: not inspect.isroutine(x)):
        if not (attr.startswith("__") or attr.endswith("__")):
            pairs.append((attr, value))
    return pairs


def get_all_methods(klass):
    """Get all method members (regular, static, class method).
    """
    if not inspect.isclass(klass):
        raise ValueError

    pairs = list()
    for attr, value in inspect.getmembers(
            klass, lambda x: inspect.isroutine(x)):
        if not (attr.startswith("__") or attr.endswith("__")):
            pairs.append((attr, value))
    return pairs


if __name__ == "__main__":
    class Base(object):
        attribute = "attribute"

        @property
        def property_method(self):
            return "property_method"

        def regular_method(self):
            return "regular_method"

        @staticmethod
        def static_method():
            return "static_method"

        @classmethod
        def class_method(cls):
            return "class_method"

    class MyClass(Base):
        pass

    def export_true_table():
        """Export value, checker function output true table.
        Help to organize thought.
        """
        import pandas as pd

        attr_value_paris = [
            ("attribute", MyClass.attribute),
            ("property_method", MyClass.property_method),
            ("regular_method", MyClass.regular_method),
            ("__dict__['static_method']", Base.__dict__["static_method"]),
            ("__dict__['class_method']", Base.__dict__["class_method"]),
            ("static_method", MyClass.static_method),
            ("class_method", MyClass.class_method),
        ]
        tester_list = [
            ("inspect.isroutine", lambda v: inspect.isroutine(v)),
            ("inspect.isfunction", lambda v: inspect.isfunction(v)),
            ("inspect.ismethod", lambda v: inspect.ismethod(v)),
            ("isinstance.property", lambda v: isinstance(v, property)),
            ("isinstance.staticmethod", lambda v: isinstance(v, staticmethod)),
            ("isinstance.classmethod", lambda v: isinstance(v, classmethod)),
        ]

        df = pd.DataFrame()
        for attr, value in attr_value_paris:
            col = list()
            for name, tester in tester_list:
                if tester(value):
                    flag = 1
                else:
                    flag = 0
                col.append(flag)
            df[attr] = col
        df.index = [name for name, _ in tester_list]

        import sys

        PY2 = sys.version_info[0] == 2
        PY3 = sys.version_info[0] == 3

        if PY2:
            fname = "PY2"
        elif PY3:
            fname = "PY3"

        writer = pd.ExcelWriter("%s.xlsx" % fname)
        df.to_excel(writer, fname, index=True)
        writer.save()

    export_true_table()

    def test_is_attribute_property_method_regular_method_static_method_class_method():
        assert is_attribute(MyClass, "attribute", MyClass.attribute)
        assert is_property_method(
            MyClass, "property_method", MyClass.property_method)
        assert is_regular_method(
            MyClass, "regular_method", MyClass.regular_method)
        assert is_static_method(
            MyClass, "static_method", MyClass.static_method)
        assert is_class_method(MyClass, "class_method", MyClass.class_method)

        attr_list = [
            (MyClass, "attribute", MyClass.attribute),
            (MyClass, "property_method", MyClass.property_method),
            (MyClass, "regular_method", MyClass.regular_method),
            (MyClass, "static_method", MyClass.static_method),
            (MyClass, "class_method", MyClass.class_method),
        ]

        checker_list = [
            is_attribute,
            is_property_method,
            is_regular_method,
            is_static_method,
            is_class_method,
        ]

        for i, pair in enumerate(attr_list):
            klass, attr, value = pair
            for j, checker in enumerate(checker_list):
                if i == j:
                    assert checker(klass, attr, value) is True
                else:
                    assert checker(klass, attr, value) is False

    test_is_attribute_property_method_regular_method_static_method_class_method()

    def test_getter():
        def items_to_keys(items):
            return set([item[0] for item in items])

        assert items_to_keys(get_attributes(MyClass)) == {"attribute"}
        assert items_to_keys(
            get_property_methods(MyClass)) == {"property_method"}
        assert items_to_keys(
            get_regular_methods(MyClass)) == {"regular_method"}
        assert items_to_keys(
            get_static_methods(MyClass)) == {"static_method"}
        assert items_to_keys(
            get_class_methods(MyClass)) == {"class_method"}

        assert items_to_keys(
            get_all_attributes(MyClass)) == {"attribute", "property_method"}
        assert items_to_keys(
            get_all_methods(MyClass)) == {"regular_method", "static_method", "class_method"}

    test_getter()
