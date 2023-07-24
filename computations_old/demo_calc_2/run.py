"""Compare TTOpt with gradient-based methods.

Please, run the following shell scripts:
cd demo_calc_2
git clone https://github.com/rfeinman/pytorch-minimize.git
cd pytorch-minimize
pip install -e .
clear && python run.py

"""

import numpy as np
import teneva
from time import perf_counter as tpc
import torch
from torchmin import minimize


from func_demo_brown import FuncDemoBrown
from ttopt import TTOpt


FUNC_NAMES = ['Ackley', 'Alpine', 'Exponential', 'Grienwank', 'Michalewicz', 'Qing', 'Rastrigin', 'Schaffer', 'Schwefel']


# Solvers:
METHODS = [
    'bfgs',
    'l-bfgs',
    'cg',
    'newton-cg',
    'newton-exact',
    'trust-ncg',
    # 'trust-krylov',  # It leads to error for the most benchmarks!
    'trust-exact',
    # 'dogleg',        # It leads to error for the most benchmarks!
]
OPT_NAMES = [
    'TTOpt',
    'BFGS',
    'L-BFGS',
    'CG', # 'Conjugate Gradient (CG)',
    'NCG', # 'Newton Conjugate Gradient (NCG)',
    'Newton', # 'Newton Exact'
    'TR NCG', # 'Trust-Region NCG',
    # 'Trust-Region GLTR (Krylov)', # It leads to error for the most benchmarks!
    'TR', # 'Trust-Region Exact',
    # 'Dogleg',                     # It leads to error for the most benchmarks!
]


def build_latex(result):
    text = ''

    for i, method in enumerate(OPT_NAMES):
        t_all = [item['t'][i] for item in result.values()]
        e_all = [item['e'][i] for item in result.values()]

        text += '\\multirow{2}{*}{\\func{' + method + '}}'

        text += '\n& $\\epsilon$ '
        for e in [item['e'][i] for item in result.values()]:
            text += f'& {e:-8.1e} '

        text += '\n\\\\'

        text += '\n& $\\tau$ '
        for t in [item['t'][i] for item in result.values()]:
            text += '& \\textit{' + f'{t:-6.2f}' + '} '

        text += ' \\\\ \\hline \n\n'

    print('\n\n Latex:\n\n' + text)


def prepare_funcs(d):
    funcs = teneva.func_demo_all(d=d, names=FUNC_NAMES)

    FUNC_NAMES.insert(2, 'Brown')
    funcs.insert(2, FuncDemoBrown(d=d))

    return funcs


def sample_init(d, a, b):
    rng = np.random.default_rng(12345)
    return torch.tensor((b - a) * rng.random(d) + a)


def run(d=10, m=int(1.E+5), p=2, q=25, r=4, reps=10):
    time_total = tpc()

    method_all = ['TTOpt'] + METHODS
    method_all = [m + ' '*max(0, 12-len(m)) for m in method_all]
    print('Method          : ' + ' | '.join(method_all))

    result = {}
    for func in prepare_funcs(d):
        t_all = []
        e_all = []

        method = 'TTOpt'
        t = tpc()
        f = func._comp
        tto = TTOpt(f, d=func.d, a=func.a, b=func.b, p=p, q=q, evals=m)
        tto.minimize(r)
        y_min = tto.y_min
        e = abs(func.y_min - y_min)
        t = tpc() - t

        t_all.append(t)
        e_all.append(e)

        for method in METHODS:
            t_cur = []
            e_cur = []

            if func.name == 'Brown' and method in ['cg']:
                t_cur = [-1]
                e_cur = [-1]
            else:
                for rep in range(reps):
                    np.random.seed(rep)
                    torch.manual_seed(rep)

                    x0 = sample_init(func.d, func.a, func.b)

                    t = tpc()
                    try:
                        res = minimize(func._calc_pt, x0, method=method,
                            max_iter=m)
                        y_min = res.fun.item()
                        e = abs(func.y_min - y_min)
                    except Exception as err:
                        print(f'Error for {method} : ', err)
                        y_min = None
                        e = -1
                    t = tpc() - t

                    t_cur.append(t)
                    e_cur.append(e)

            t_all.append(np.mean(t_cur))
            e_all.append(np.mean(e_cur))

        result[func.name] = {'t': t_all, 'e': e_all}

        name = func.name + ' '*max(0, 12-len(func.name))
        t_all = ' | '.join([f'{t:-12.4f}' for t in t_all])
        e_all = ' | '.join([f'{e:-12.1e}' for e in e_all])
        print()
        print(name + '(t) : ' + t_all)
        print(name + '(e) : ' + e_all)

    build_latex(result)

    time_total = tpc() - time_total
    print(f'\n\nDONE | Time: {time_total:-8.4f}')


if __name__ == '__main__':
    run()
