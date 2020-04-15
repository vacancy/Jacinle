# Jacinle
Jacinle is a personal python toolbox developed by Jiayuan Mao.
It contains a range of utility functions for python development,
including project configuration, file IO, image processing, inter-process communication, etc.

## Installation and Usage
Step 1: clone the Jacinle package, and add the bin path to your PATH environment.
```
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```

Step 2: install the required third-party packages, see requirements.txt for a list.

Step 3: you are all set. Use `jac-run xxx.py` to replace `python3 xxx.py`. You can also use the
`jac-crun <gpu_ids> xxx.py` to set the gpus you want to use. Here, `<gpu_ids>` is a comma-separated list of
gpu ids, following the convension of `CUDA_VISIBLE_DEVICES`.


## Advanced Usage

