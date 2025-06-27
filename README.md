phaser: The weapon of choice for ptychographic reconstructions
---
[![][ci-badge]][ci-url] [![][commit-badge]][commit-url] [![][docs-dev-badge]][docs-dev-url] [![][binder-badge]][binder-url]


`phaser` is a fully-featured package for multislice electron ptychography.
See our [arXiv paper](https://arxiv.org/abs/2505.14372) for more details.

## Features

- ePIE, LSQ-MLs, and gradient descent algorithms
- [`numpy`][numpy], [`cupy`][cupy], and [`jax`][jax] backends.
- Single and multislice ptychography
- Multiple incoherent probe modes
- Probe position correction
- Upsampled (sPIE) and segmented ptychography (work in progress)
- Adaptive propagator correction (contributed by M Zhu)

## Documentation

Documentation on `phaser` can be found here: https://hexane360.github.io/phaser/dev/

Documentation is still very much a work in progress, so please feel free to open an issue or email me if you have any questions!

## Installation

To install, first clone the repository from github.
This can be done from GitHub Desktop, or from the git command line:

```sh
$ git clone https://github.com/hexane360/phaser
# enter phaser directory
$ cd phaser
```

We recommend using a conda environment or Python virtual environment to keep things clean, although this is not mandatory.

`phaser` supports multiple computational backends. The simplest (and slowest) is `numpy`. `cupy` can be used for CUDA-accelerated. `jax` supports CPU and GPU acceleration, and is the only backend which supports the gradient descent engine.
If you're unsure what engines to use, we recommend installing the `jax` engine.

If you're using [`cupy`][cupy] or [`jax`][jax] with a GPU, start by following the installation instructions for those packages.
Jax can be installed with or without CUDA support, if you're using CUDA make sure you install the correct version. Currently, Jax does not support CUDA on Windows.

Before moving on to installing `phaser`, make sure those packages you've installed work:
```sh
$ python
>>> import jax
>>> jax.default_backend()
'gpu'  # should be 'gpu' on cuda, 'cpu' otherwise
>>> jax.numpy.array([1, 2, 3, 4]) + 1   # test a basic operation
Array([2, 3, 4, 5], dtype=int32)

>>> import cupy
>>> cupy.array([1, 2, 3, 4]) + 1
array([2, 3, 4, 5])
```

Then, install `phaser` using `pip`:

```sh
$ python -m pip install -e .
```

For the jax or cupy backend, or for the optional webserver, install with the corresponding options:

```sh
$ python -m pip install -e ".[jax,cupy12,web]" # for the 'jax', 'cupy12', and 'web' options
```

Depending on your command line, you may need to put double quotes around the options (as shown).

Here are the supported installation options:

- `jax`: For the [`jax`][jax] backend (required for the gradient descent engine)
- `cupy11`: `cupy` for CUDA toolkit 11.x
- `cupy12`: `cupy` for CUDA toolkit 12.x
- `web`: For the web interface

For [Optuna](https://optuna.org/) hyperoptimization, install it as well:

```sh
$ pip install optuna
```

## Running

After installation, the `phaser` command should be available. Phaser can be run from the command line, or through a job server.

To run a single reconstruction on the command line, call `phaser run <file>`, where `file` is the path to a reconstruction plan file.

To run the webserver, call `phaser serve`. By default, the server serves on https://localhost:5050/, so navigate there in a web browser. The server interface can be used to start workers and schedule reconstruction jobs.

To run a worker, call `phaser worker <url>`, where `url` is the URL of a running job server.

## Sample data & Examples

Sample data can be downloaded from the following dropbox link: https://www.dropbox.com/scl/fo/txm3k88ubrzvt541v23ir/AL-l_m6VnGlFxzHWZSSc0TA?rlkey=8qxtwnc8cwhpff6jpr5s40y6i&st=x9pbwke0&dl=0

Copy the `sample_data` directory into the root code folder.

```sh
$ curl --output sample_data.zip -L 'https://www.dropbox.com/scl/fo/txm3k88ubrzvt541v23ir/AL-l_m6VnGlFxzHWZSSc0TA?rlkey=8qxtwnc8cwhpff6jpr5s40y6i&st=x9pbwke0&dl=1'
$ unzip sample_data.zip -x / -d sample_data
```

Sample data includes simulated and experimental MoS2 data, simulated and experimental Si data, and experimental PrScO3 data.
After the data is downloaded, any of the example reconstructions can be run as `phaser run examples/mos2_grad.yaml` (for example).

## Alternatives

Other notable ptychography packages:

 - [`fold_slice`](https://github.com/yijiang1/fold_slice) branch of [PtychoShelves](https://www.psi.ch/en/sls/csaxs/software#coming-soon-ptychoshelves-a-versatile-high-level-framework-for-high-performance-analysis-of)
 - [`py4DSTEM`](https://github.com/py4dstem/py4DSTEM)
 - [`PtyLab.m/py/jl`](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026)
 - [PtyPy](https://ptycho.github.io/ptypy/)
 - [PtyRAD](https://github.com/chiahao3/ptyrad)

[numpy]: https://numpy.org/
[cupy]: https://cupy.dev/
[jax]: https://docs.jax.dev/en/latest/

[ci-badge]: https://github.com/hexane360/phaser/workflows/Tests/badge.svg
[ci-url]: https://github.com/hexane360/phaser/actions/workflows/ci.yaml
[docs-dev-badge]: https://img.shields.io/badge/docs-dev-blue
[docs-dev-url]: https://hexane360.github.io/phaser/dev/
[commit-badge]: https://img.shields.io/github/last-commit/hexane360/phaser
[commit-url]: https://github.com/hexane360/phaser/commits
[binder-badge]: https://mybinder.org/badge_logo.svg
[binder-url]: https://mybinder.org/v2/gh/hexane360/phaser/HEAD
