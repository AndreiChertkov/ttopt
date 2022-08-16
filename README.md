# ttopt

> Gradient-free optimization method for multivariable functions based on the low rank tensor train (TT) format and maximal-volume principle.


## Installation

1. Install [python](https://www.python.org) (version >= 3.7; you may use [anaconda](https://www.anaconda.com) package manager);
2. Install basic dependencies:
    ```bash
    pip install numpy cython scipy
    ```
3. Install additional dependency [maxvolpy](https://bitbucket.org/muxas/maxvolpy/src/master/) with effective realization of `maxvol` algorithm:
    ```bash
    pip install maxvolpy
    ```
    > Note that [teneva](https://github.com/AndreiChertkov/teneva) python package may be used instead (`pip install teneva`)
4. Install dependencies for demo calculations (it is optional):
    ```bash
    pip install matplotlib cma nevergrad pyproj teneva
    ```
5. Install the `ttopt` package:
    ```bash
    python setup.py install
    ```
    > You may also use the pip for installation, i.e., `pip install ttopt`


## Examples

The demo-scripts with detailed comments are collected in the folder `demo`:

- `base.py` - we find the minimum for the 10-dimensional function with vectorized input;
- `qtt.py` - we do almost the same as in the `base.py` script, but use the QTT-based approach (note that results are much more better then in the `base.py` example);
- `qtt_100d.py` - we do almost the same as in the `qtt.py` script, but approximate the 100-dimensional function;
- `vect.py` - we find the minimum for the simple analytic function with "simple input" (the function is not vectorized);
- `cache.py` - we find the minimum for the simple analytic function to demonstrate the usage of cache;
- `tensor.py` - in this example we find the minimum for the multidimensional array/tensor (i.e., discrete function).
- `tensor_init_spec` - we do almost the same as in the `tensor.py` script, but use special method of initialization (instead of a random tensor, we select a set of starting multi-indices for the search).

> The folder `check` contains a number of additional calculations related to the study of the algorithm and the evaluation of its performance. A more accurate assessment of the accuracy and efficiency of the TTOpt approach is given in the folder `demo_calc` (see description below).


## Calculations for benchmarks

The scripts for comparison of our approach with baselines (ES algorithms and some baselines from the `nevergrad` package) for the analytical benchmark functions are located in the folder `demo_calc`. To run calculations, you can proceed as follows `python demo_calc/run.py --KIND`. Possible values for `KIND`: `comp` - compare different solvers; `dims` - check dependency on dimension number; `iter` - check dependency on number of calls for the target function; `quan` - check effect of the QTT-usage; `rank` - check dependency on the rank; `show` - show results of the previous calculations.

> All results will be collected in the folders `demo_calc/res_data` (saved results in the pickle format), `demo_calc/res_logs` (text files with logs) and `demo_calc/res_plot` (figures with results).

To reproduce the results from the paper (it is currently in the process of being published), run the following scripts from the root folder of the package:
1. Run `python demo_calc/run.py -d 10 -p 2 -q 25 -r 4 --evals 1.E+5 --reps 10 --kind comp`;
2. Run `python demo_calc/run.py -p 2 -q 25 -r 4 --reps 1 --kind dims`;
3. Run `python demo_calc/run.py -d 10 -p 2 -q 25 -r 4 --reps 10 --kind iter`;
4. Run `python demo_calc/run.py -d 10 -r 4 --evals 1.E+5 --reps 10 --kind quan`;
5. Run `python demo_calc/run.py -d 10 -p 2 -q 25 --evals 1.E+5 --reps 10 --kind rank`;
6. Run `python demo_calc/run.py -d 10 --kind show`. The results will be saved to the `demo_calc/res_logs` and `demo_calc/res_plot` folders.

> Additional comparison with gradient-based methods is presented in the `demo_calc_2` folder.

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
    title     = {TTOpt: A Maximum Volume Quantized Tensor Train-based Optimization and its Application to Reinforcement Learning},
    journal   = {ArXiv},
    volume    = {abs/2205.00293},
    doi       = {10.48550/ARXIV.2205.00293},
    url       = {https://arxiv.org/abs/2205.00293}
}
```
