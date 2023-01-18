# Jacinle

Jacinle is a personal python toolbox.
It contains a range of utility functions for python development,
including project configuration, file IO, image processing, inter-process communication, etc.

[[Website]](http://jacinle.jiayuanm.com/)
[[Examples]] (https://github.com/vacancy/Jacinle/tree/master/examples)
[[Jacinle References]](http://jacinle.jiayuanm.com/reference/jacinle.html)
[[JacLearn References]](http://jacinle.jiayuanm.com/reference/jaclearn.html)
[[JacTorch References]](http://jacinle.jiayuanm.com/reference/jactorch.html)


## Installation

Clone the Jacinle package (be sure to clone all submodules), and add the bin path to your PATH environment.

```
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```

Optionally, you may need to install third-party packages specified in `requirements.txt`

## Command Line

1. `jac-run xxx.py`

    Jacinle comes with a command line to replace the `python` command: `jac-run`. In short, this command
    will automatically add the Jacinle packages into `PYTHONPATH`, as well as adding a few vendor Python packages
    into `PYTHONPATH` (for example, [JacMLDash](https://github.com/vacancy/JacMLDash)). Using this command
    to replace `python xxx.py` is the best practice to manage dependencies.

    Furthremore, this command also supports a configuration file specific to projects. The command will search for
    a configuration file named `jacinle.yml` in the current working directory and its parent directories. This file
    specifies additional environmental variables to add, for example.

    ```
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
    ```

2. `jac-crun <gpu_ids> xxx.py`

    The same as `jac-run`, but takes an additional argument, which is a comma-separated list of gpu ids,
    following the convension of `CUDA_VISIBLE_DEVICES`.

3. `jac-debug xxx.py`

    The same as `jac-run`, but sets the environment variable `JAC_DEBUG=1` before running the command.
    By default, in the debug mode, an `ipdb` interface will be started when an exception is raised.

4. `jac-cdebug <gpu_ids> xxx.py`

    The combined `jac-debug` and `jac-crun`.

5. `jac-update`

    Update the Jacinle package (and all dependencies inside `vendors/`).

6. `jac-inspect-file xxx.json yyy.pkl`

    Start an IPython interface and loads all files in the argument list. The content of the files can be accessed via `f1`, `f2`, ...

## Python Libraries

Jacinle contains a collection of useful packages. Here is a list of commonly used packages, with links to the documentation.

- [`jacinle.*`](https://jacinle.jiayuanm.com/reference/jacinle.io.html): frequently used utility functions, such as `jacinle.JacArgumentParser`, `jacinle.TQDMPool`, `jacinle.get_logger`, `jacinle.cond_with`, etc.
- [`jacinle.io.*`](https://jacinle.jiayuanm.com/reference/jacinle.io.html): IO functions. Two of the mostly used ones are: `jacinle.io.load(filename)` and `jacinle.io.dump(filename, obj)`
- [`jacinle.random.*`](https://jacinle.jiayuanm.com/reference/jacinle.random.html): almost the same as `numpy.random.*`, but with a few additional utility functions and RNG state management functions.
- [`jacinle.web.*`](https://jacinle.jiayuanm.com/reference/jacinle.web.html): the old `jacweb` package, which is a customized wrapper around the [tornado](https://www.tornadoweb.org/en/stable/) web server.
- [`jaclearn.*`](https://jacinle.jiayuanm.com/reference/jaclearn.html): machine learning modules.
- [`jactorch.*`](https://jacinle.jiayuanm.com/reference/jactorch.html): a collection of PyTorch functions in addition to the `torch.*` functions.

