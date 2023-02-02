# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(dirname(__file__))))


# -- Project information -----------------------------------------------------

project = 'Jacinle'
copyright = '2022, Jiayuan Mao'
author = 'Jiayuan Mao'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'nbsphinx',
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# autosummary
# autosummary_generate = True  # Turn on sphinx.ext.autosummary

# autodoc
autodoc_mock_imports = ['pyrealsense2', 'tensorflow', 'patch_match', 'pygco', 'pygco_inpaint', 'ppaquette_gym_super_mario', 'mujoco_py']
autodoc_inherit_docstrings = True
autodoc_class_signature = 'separated'
# autodoc_member_order = 'bysource'
autodoc_member_order = 'groupwise'
autodoc_typehints = 'both'
autodoc_typehints_format = 'short'
autodoc_typehints_description_target = 'all'
autodoc_preserve_defaults = True
# We will use autosummary to generate per-function and per-class docs.
# autodoc_default_options = {'members': True, 'undoc-members': True, 'inherited-members': True, 'show-inheritance': True}

autoclass_content = "class"

add_module_names = False  # Remove namespaces from class/method signatures
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
autodoc_typehints = "description"  # Sphinx-native method. Not as good as sphinx_autodoc_typehints

# napoleaon
napolean_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------

# Pydata theme
html_theme = "pydata_sphinx_theme"
# html_logo = "_static/logo-company.png"
html_theme_options = {
    'show_prev_next': False,
    "show_nav_level": 0
}
html_css_files = ['custom.css']
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

