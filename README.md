# Steerable PDO Experiments
**[Library code](https://github.com/ejnnr/steerable_pdos)** | **[Paper](https://arxiv.org/abs/2106.10163)** | **[Original library](https://github.com/QUVA-Lab/e2cnn)**

This repository contains the experiments for our paper on [steerable PDOs](https://arxiv.org/abs/2106.10163).
It makes use of our [extension of the *e2cnn* library](https://github.com/ejnnr/steerable_pdos).
Parts of the code are also reused from the [experiments code for the *e2cnn* paper](https://github.com/QUVA-Lab/e2cnn_experiments).

## Setup
We use [`poetry`](https://python-poetry.org/) to manage package dependencies
because this makes it easy to lock the precise package versions with which
we tested the code. If you have `poetry` installed, just run `./install.sh`,
which will create a new virtual environment and install all dependencies
into this environment. You can activate it with `poetry shell`.

**Note:** If you want to recreate the figures from the paper, using
`figures.py`, you also need to install `seaborn`, which is not included in
the `poetry` environment.

Of course you don't need to use `poetry`, you can also install the requirements
yourself using `pip` or `conda`. You can find a list of required packages
in `pyproject.toml`. In addition to the once listed there, you will need
to install the [`RBF`](https://github.com/treverhines/RBF) package.
Finally, run
```
pip install git+https://github.com/ejnnr/steerable_pdos@pdo_econv
```
to install the version of the *e2cnn* library that we need for these experiments.

## Running experiments
### MNIST-rot
The entry point into the MNIST-rot experiments is `python main.py`.
The simplest way to use this command is
```
python main.py +experiment=<experiment name>
```
where `<experiment name` can be any combination of `{diffop,kernel,vanilla}_{3x3,5x5}`.
For example, to reproduce our MNIST-rot result for 3x3 kernels, run `python main.py +experiment=kernel_3x3`.

For differential operators, the default is FD discretization. If you want to use Gaussians,
use `+model.smoothing=<standard deviation>`, e.g.
```
python main.py +experiment=diffop_5x5 +model.smoothing=1.3
```
(we used 1.3 for 5x5 kernels and 1 for 3x3 kernels). For RBF-FD, use `+model.rbffd=true` instead.

To reproduce the restriction models (which start with D_N equivariance and restrict to C_N), use
`+model.flip=true +model.restriction_layer=6` (this restricts for the 6th layer, you can change
that number). For the quotient experiments, use `+model.quotient=true`.
To use SO(2) irreps instead of regular representations, use `+model.group_order=-N`
where `N` is the maximum irrep frequency you want to use (`N = 3` is reasonable).
Note the minus sign; without it, this would use `C_N` as the symmetry group.

Finally, our code also allows you to exactly imitate the [PDO-eConv](https://arxiv.org/abs/2007.10408) basis.
To do so, add the `model.pdo_econv` option, i.e.
```
python main.py +experiment=diffop_5x5 +model.pdo_econv=1
```
You can combine this with `+model.smoothing=...` to use Gaussian discretization.
But in general, the PDO-eConv basis is less flexible and thus cannot be combined
with all of the options described above. It also currently only supports 5x5 kernels.

`main.py` has many other options, so there are many architecture and hyperparameter
choices that you can easily modify. For example, the following command illustrates
a few of them:
```
python main.py +experiment=diffop_5x5 \
    trainer.max_epochs=50 \
    data.batch_size=32 \
    model.learning_rate=0.001 \
    model.maximum_order=2 \
    model.weight_decay=1e-4 \
    model.fc_dropout=0.2 \
    model.lr_decay=0.9 \
    model.optimizer=sgd \
    model.channels=\[20, 30, 40, 40, 50, 70\]
```
In the `config/` directory, you can see some more option, as well as even more
in `diffop_experiments/model.py`. Don't hesitate to [contact me](mailto:erik@ejenner.com)
or file a Github issue if you'd like to try something not mentioned here, maybe it's
already implemented. 

Some further options that you will probably want to set:
- `+trainer.gpus=1` to use the GPU
- `data.num_workers=<number of workers>` to use multiple workers and speed up training
- `seed=<random seed>`: a seed will always be used, by default it is 0. So if you want multiple runs, change the seed!
- `dir.run=<directory>` to save the logs for that run in `logs/<directory>`.
  For example, something like `dir.run=diffop/3x3/gaussian/<seed>` may be useful.
  By default, a directory based on the current date and time is created for each run.

### STL-10
For the STL-10 experiments, we simply reused the code for the
[experiments for the *e2cnn* paper](https://github.com/QUVA-Lab/e2cnn_experiments),
with only very small additions to support Steerable PDOs. These experiments
therefore have different entrypoints, see the original repo for details.

To run exactly those experiments that we ran for our paper, you can use `./run_stl.sh`.
This will do six runs for each of the eight models we consider, and will probably take
on the order of 300h on a GPU (depending on your exact system of course).
To have more flexibility, e.g. for multiple parallel runs, you can see the individual
commands for each model in `./run_stl.sh`.

## Figures
You can run `python figures.py` to reproduce the figures from our paper. You will have
to install `seaborn` first. The figures will be saved into the `fig` folder.

## Cite

The original *e2cnn* library and parts of the experiments code in this repository
were developed as part of the
[General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251) paper.
The extension of the library to steerable PDOs and other parts of the experiment code
were written for our [steerable PDO paper](https://arxiv.org/abs/2106.10163).
Please cite these papers if you find the code in this repository useful for your own work:

```
@inproceedings{e2cnn,
    title={{General E(2)-Equivariant Steerable CNNs}},
    author={Weiler, Maurice and Cesa, Gabriele},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2019},
}
@misc{jenner2021steerable,
    title={Steerable Partial Differential Operators for Equivariant Neural Networks}, 
    author={Erik Jenner and Maurice Weiler},
    year={2021},
    eprint={2106.10163},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

All code in this repository is distributed under the BSD Clear license. See LICENSE file.
