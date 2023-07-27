# ttopt


## Description

Gradient-free optimization method for multivariable functions based on the low rank tensor train (TT) format and maximal-volume principle.

> Please, see also our software product [teneva](https://github.com/AndreiChertkov/teneva) which provides a very compact implementation of basic operations in the TT-format.


## Installation

You can install the `ttopt` package for `python >= 3.7` with pip:
```bash
pip install ttopt==0.6.2
```


## Examples

The demo-scripts with detailed comments are collected in the folder `demo`:

- `base.py` - we find the minimum for the 10-dimensional function with vectorized input;
- `qtt.py` - we do almost the same as in the `base.py` script, but use the QTT-based approach (note that results are much more better then in the `base.py` example);
- `qtt_max.py` - we do almost the same as in the `qtt.py`, but consider the maximization task;
- `qtt_100d.py` - we do almost the same as in the `qtt.py` script, but approximate the 100-dimensional function;
- `vect.py` - we find the minimum for the simple analytic function with "simple input" (the function is not vectorized);
- `cache.py` - we find the minimum for the simple analytic function to demonstrate the usage of the cache;
- `tensor.py` - in this example we find the minimum for the multidimensional array/tensor (i.e., discrete function);
- `tensor_init` - we do almost the same as in the `tensor.py` script, but we use special method of initialization (instead of a random tensor, we select a set of starting multi-indices for the search).


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)
- [Roman Schutski](https://github.com/Qbit-)
- [Konstantin Sozykin](https://github.com/gogolgrind)


## Citation

If you find this approach and/or code useful in your research, please consider citing:

```bibtex
@article{sozykin2022ttopt,
    author    = {Sozykin, Konstantin and Chertkov, Andrei and Schutski, Roman and Phan, Anh-Huy and Cichocki, Andrzej and Oseledets, Ivan},
    year      = {2022},
    title     = {{TTOpt}: {A} maximum volume quantized tensor train-based optimization and its application to reinforcement learning},
    journal   = {Advances in Neural Information Processing Systems},
    volume    = {35},
    pages     = {26052--26065},
    url       = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/a730abbcd6cf4a371ca9545db5922442-Abstract-Conference.html}
}
```

> Please, note that the calculations presented in this paper correspond to version `<0.5.0` of the `ttopt` package (and to very old version of the `teneva` package), to run the calculations, please use the appropriate version. In the new versions `>=0.6.0`, we have removed all the corresponding folders in the folder `computations_old`. In the future, we will try to update the interface of these experiments.


---


> ✭__🚂  The stars that you give to **ttopt**, motivate us to develop faster and add new interesting features to the code 😃
