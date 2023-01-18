.. Jacinle documentation master file, created by
   sphinx-quickstart on Sat Apr 21 16:51:41 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :hidden:

   Home Page <self>
   Jacinle Reference <reference/jacinle>
   JacLearn Reference <reference/jaclearn>
   JacTorch Reference <reference/jactorch>


============
Jacinle
============

Jacinle is a personal python toolbox.
It contains a range of utility functions for python development,
including project configuration, file IO, image processing, inter-process communication, etc.

- Website: https://jacinle.jiayuanm.com
- Github: https://github.com/vacancy/Jacinle
- Examples: https://github.com/vacancy/Jacinle/tree/master/examples
- Jacinle Reference: http://jacinle.jiayuanm.com/reference/jacinle.html
- JacLearn Reference: http://jacinle.jiayuanm.com/reference/jaclearn.html
- JacTorch Reference: http://jacinle.jiayuanm.com/reference/jactorch.html

Installation
============

Clone the Jacinle package (be sure to clone all submodules), and add the bin path to your PATH environment.

.. code-block:: bash

  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<path_to_jacinle>/bin:$PATH

Optionally, you may need to install third-party packages specified in :code:`requirements.txt`

Command Line
============

1. :code:`jac-run xxx.py` Jacinle comes with a command line to replace the :code:`python` command.

In short, this command will automatically add the Jacinle packages into :code:`PYTHONPATH`, as well as adding a few vendor Python packages
into :code:`PYTHONPATH` (for example, (https://github.com/vacancy/JacMLDash)). Using this command
to replace :code:`python xxx.py` is the best practice to manage dependencies.

Furthremore, this command also supports a configuration file specific to projects. The command will search for
a configuration file named :code:`jacinle.yml` in the current working directory and its parent directories. This file
specifies additional environmental variables to add, for example.

.. code-block:: yaml

  project_root: true  # tell the script that the folder containing this file is the root of a project. The directory will be added to PYTHONPATH.
  system:
      envs:
          CUDA_HOME: /usr/local/cuda-10.0  # set needed environment variables here.
  path:
      bin:  # will be prepended to $PATH
          /usr/local/bin
      python:  # will be prepended to $PYTHONPATH
          /Users/jiayuanm/opt/my_python_lib
  vendors:  # load additional Python packages (root paths will be added to PYTHONPATH)
      pybullet_tools:
          root: /Users/jiayuanm/opt/pybullet/utils
      alfred:
          root: /Users/jiayuanm/opt/alfred

2. :code:`jac-crun <gpu_ids> xxx.py` The same as :code:`jac-run`, but takes an additional argument, which is a comma-separated list of gpu ids, following the convension of :code:`CUDA_VISIBLE_DEVICES`.

3. :code:`jac-debug xxx.py` The same as :code:`jac-run`, but sets the environment variable :code:`JAC_DEBUG=1` before running the command. By default, in the debug mode, an :code:`ipdb` interface will be started when an exception is raised.

4. :code:`jac-cdebug <gpu_ids> xxx.py` The combined :code:`jac-debug` and :code:`jac-crun`.

5. :code:`jac-update` Update the Jacinle package (and all dependencies inside :code:`vendors/`).

6. :code:`jac-inspect-file xxx.json yyy.pkl` Start an IPython interface and loads all files in the argument list. The content of the files can be accessed via :code:`f1`, :code:`f2`, ...

Python Libraries
================

Jacinle contains a collection of useful packages. Here is a list of commonly used packages, with links to the documentation.

- :code:`jacinle.*`        (https://jacinle.jiayuanm.com/reference/jacinle.io.html): frequently used utility functions, such as :code:`jacinle.JacArgumentParser`, :code:`jacinle.TQDMPool`, :code:`jacinle.get_logger`, :code:`jacinle.cond_with`, etc.
- :code:`jacinle.io.*`     (https://jacinle.jiayuanm.com/reference/jacinle.io.html): IO functions. Two of the mostly used ones are: :code:`jacinle.io.load(filename)` and :code:`jacinle.io.dump(filename, obj)`
- :code:`jacinle.random.*` (https://jacinle.jiayuanm.com/reference/jacinle.random.html): almost the same as :code:`numpy.random.*`, but with a few additional utility functions and RNG state management functions.
- :code:`jacinle.web.*`    (https://jacinle.jiayuanm.com/reference/jacinle.web.html): the old :code:`jacweb` package, which is a customized wrapper around the [tornado](https://www.tornadoweb.org/en/stable/) web server.
- :code:`jaclearn.*`       (https://jacinle.jiayuanm.com/reference/jaclearn.html): machine learning modules.
- :code:`jactorch.*`       (https://jacinle.jiayuanm.com/reference/jactorch.html): a collection of PyTorch functions in addition to the :code:`torch.*` functions.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

