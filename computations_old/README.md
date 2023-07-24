# Evaluation of the TTOpt efficiency [OUTDATED!]

> Please, note that the calculations presented in this folder correspond to version `<0.5.0` of the `ttopt` package (and to very old version of the `teneva` package), to run the calculations, please use the appropriate version. In the new versions `>=0.6.0`, we have removed all the corresponding folders in the folder "computations_old". In the future, we will try to update the interface of these experiments.


## Calculations for benchmarks [OUTDATED!]

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


## TTOpt for RL [MAYBE OUTDATED!]

Please, see the notebook `TTopt_InvertedPendulum_demo.ipynb` with the minimalistic example of using **TTOpt** for optimal on-policy search in Reinforcement Learning problems.

> This particular example is recommended to be run in Colab, otherwise you may need docker or root access to install mujoco (see https://github.com/openai/mujoco-py).
