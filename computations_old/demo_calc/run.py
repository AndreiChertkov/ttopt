"""Investigating TTOpt performance for analytical benchmark functions."""
import argparse
import numpy as np
import pickle
import random
import sys


import matplotlib.pyplot as plt
params = {
    'text.usetex' : False,
    'font.size' : 36,
    'legend.fancybox':True,
    'legend.loc' : 'best',
    'legend.framealpha': 0.9,
    "legend.fontsize" : 27}
plt.rcParams.update(params)


sys.path.append('./demo_calc/opt')
from opt_tt_opt import OptTTOpt
from opt_ga import OptGA
from opt_es import OptES
from opt_es_cma import OptESCMA
from opt_de import OptDE
from opt_nb import OptNB
from opt_pso import OptPSO


from teneva import func_demo_all


# Minimizer classes:
OPTS = [OptTTOpt, OptGA, OptES, OptESCMA, OptDE, OptNB, OptPSO]


# Minimizer names:
OPT_NAMES = {
    'TTOpt': 'TTOpt',
    'GA': 'GA',
    'ES-OpenAI': 'openES',
    'ES-CMA': 'cmaES',
    'DE': 'DE',
    'NB': 'NB',
    'PSO': 'PSO',
}


# Function names and possible dimensions (set "True" if works for any):
FUNCS = {
    'Ackley': True,
    'Alpine': True,
    'Brown': True,
    'Exponential': True,
    'Grienwank': True,
    'Michalewicz': [10],
    'Qing': True,
    'Rastrigin': True,
    'Schaffer': True,
    'Schwefel': True,
}


# List of dimensions to check the TTOpt for multi-dim case:
D_LIST = [10, 50, 100, 500]

# List of ranks to check dependency of TTOpt on rank:
R_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# List of grid sizes to check QTT-effect (should be power of 2 and 4):
N_LIST = [2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20]


# List of numbers of function calls to check the dependency:
EVALS_LIST = [1.E+4, 5.E+4, 1.E+5, 5.E+5, 1.E+6, 5.E+6, 1.E+7]


# Population size for genetic-based algorithms:
GA_POPSIZE = 255


def get_funcs(d):
    names = []
    for name, dims in FUNCS.items():
        if isinstance(dims, list):
            if not d in dims:
                continue
        elif not dims:
            continue
        names.append(name)
    return func_demo_all(d, names=names)


def get_opt(func, opt_class, n, p, q, r, evals, with_log=False):
    opt = opt_class(func.get_f_poi, func.d, func.a, func.b,
        func.x_min, func.y_min, verb=with_log)

    if opt.name == 'TTOpt':
        opt.prep(n, p, q, r, evals)
    elif opt.name in ['DE', 'NB', 'PSO']:
        opt.prep(evals)
    else:
        opt.prep(popsize=GA_POPSIZE, iters=evals/GA_POPSIZE)

    return opt


def load(d, name, kind):
    fpath = f'./demo_calc/res_data/{name}_{kind}_{d}dim.pickle'
    try:
        with open(fpath, 'rb') as f:
            res = pickle.load(f)
    except Exception as e:
        res = None

    return res


def log(text, d, name, kind, is_init=False):
    print(text)

    fpath = f'./demo_calc/res_logs/{name}_{kind}_{d}dim.txt'
    with open(fpath, 'w' if is_init else 'a') as f:
        f.write(text + '\n')


def run_comp(d, p, q, r, evals, reps=1, name='calc1', with_log=False):
    """Compare different methods for benchmark analytic functions."""
    log(f'', d, name, 'comp', is_init=True)
    res = {}

    for func in get_funcs(d):
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'comp')
        res[func.name] = {}

        for opt_class in OPTS:
            opt = get_opt(func, opt_class, None, p, q, r, evals, with_log)
            res[func.name][opt.name] = solve(opt, d, name, 'comp', reps)

            save(res, d, name, 'comp')

        log('\n\n', d, name, 'comp')


def run_dims(p, q, r, reps=1, name='calc1', with_log=False, evals_par=1.E+4):
    """Solve for different dimension numbers."""
    d0 = D_LIST[-1]

    log(f'', d0, name, 'dims', is_init=True)
    res = {}

    for d in D_LIST:
        evals = int(evals_par * d)
        res[d] = {}

        for func in get_funcs(d):
            log(f'--- Minimize function {func.name}-{d}dim', d0, name, 'dims')

            opt = get_opt(func, OptTTOpt, None, p, q, r, evals, with_log)

            res[d][func.name] = solve(opt, d, name, 'dims', reps, d0)

            save(res, d0, name, 'dims')

            log('', d0, name, 'dims')


def run_iter(d, p, q, r, reps=1, name='calc1', with_log=False):
    """Check dependency of TTOpt on evals for benchmark analytic functions."""
    log(f'', d, name, 'iter', is_init=True)
    res = {}

    for func in get_funcs(d):
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'iter')
        res[func.name] = []

        for evals in EVALS_LIST:
            opt = get_opt(func, OptTTOpt, None, p, q, r, evals, with_log)
            res[func.name].append(solve(opt, d, name, 'iter', reps))

            save(res, d, name, 'iter')

        log('\n\n', d, name, 'iter')


def run_quan(d, r, evals, reps=1, name='calc1', with_log=False):
    """Check effect of QTT-based approach for benchmark analytic functions."""
    log(f'', d, name, 'quan', is_init=True)
    res = {}

    for func in get_funcs(d):
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'quan')

        res[func.name] = {'q0': [], 'q2': [], 'q4': []}

        for n in N_LIST:
            n = int(n)
            q2 = int(np.log2(n))
            q4 = int(q2 / 2)

            if 2**q2 != n or 4**q4 != n:
                raise ValueError(f'Invalid grid size "{n}"')

            opt = get_opt(func, OptTTOpt, n, None, None, r, evals, with_log)
            res[func.name]['q0'].append(solve(opt, d, name, 'quan', reps))

            opt = get_opt(func, OptTTOpt, None, 2, q2, r, evals, with_log)
            res[func.name]['q2'].append(solve(opt, d, name, 'quan', reps))

            opt = get_opt(func, OptTTOpt, None, 4, q4, r, evals, with_log)
            res[func.name]['q4'].append(solve(opt, d, name, 'quan', reps))

            save(res, d, name, 'quan')

        log('\n\n', d, name, 'quan')


def run_rank(d, p, q, evals, reps=1, name='calc1', with_log=False):
    """Check dependency of TTOpt on rank for benchmark analytic functions."""
    log(f'', d, name, 'rank', is_init=True)
    res = {}
    for func in get_funcs(d):
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'rank')
        res[func.name] = []

        for r in R_LIST:
            opt = get_opt(func, OptTTOpt, None, p, q, r, evals, with_log)
            res[func.name].append(solve(opt, d, name, 'rank', reps))

            save(res, d, name, 'rank')

        log('\n\n', d, name, 'rank')


def run_show(d, name='calc1'):
    """Show results of the previous calculations."""
    log(f'', d, name, 'show', is_init=True)
    run_show_comp(d, name)
    run_show_dims(d, D_LIST[-1], name)
    run_show_iter(d, name)
    run_show_quan(d, name)
    run_show_rank(d, name)


def run_show_comp(d, name='calc1'):
    """Show results of the previous calculations for "comp"."""
    res = load(d, name, 'comp')
    if res is None:
        log('>>> Comp-result is not available\n\n', d, name, 'show')
        return

    text = '>>> Comp-result (part of latex table): \n\n'

    text += '% ------ AUTO CODE START\n\n'

    for name_opt, name_opt_text in OPT_NAMES.items():
        text += '\\multirow{2}{*}{\\func{' + name_opt_text + '}}'

        text += '\n& $\\epsilon$ '
        for func in get_funcs(d):
            v = res[func.name][name_opt]['e']
            vals = [res[func.name][nm]['e']
                for nm in OPT_NAMES.keys() if nm != name_opt]
            if v <= np.min(vals):
                text += '& \\textbf{' + f'{v:-8.1e}' + '} '
            else:
                text += f'& {v:-8.1e} '

        text += '\n\\\\'

        text += '\n& $\\tau$ '
        for func in get_funcs(d):
            v = res[func.name][name_opt]['t']
            text += '& \\textit{' + f'{v:-6.2f}' + '} '

        text += ' \\\\ \\hline \n\n'

    text += '% ------ AUTO CODE END\n\n'

    log(text, d, name, 'show')


def run_show_dims(d0, d_max, name='calc1'):
    res = load(d_max, name, 'dims')
    if res is None:
        log('>>> Dims-result is not available', d0, name, 'show')
        return

    text = '>>> Dims-result: \n\n'

    text += '% ------ AUTO CODE START\n'

    for i, name_func in enumerate(res[d_max].keys(), 1):
        text += '\\multirow{2}{*}{\\emph{F' + str(i) + '}}\n'

        text += '& $\\epsilon$ \n'
        for d in D_LIST:
            item = res[d][name_func]
            e = item['e']
            text += '& ' + f'{e:-8.1e}' + '\n'

        text += '\\\\ \n'

        text += '& $\\tau$ \n'
        for d in D_LIST:
            item = res[d][name_func]
            t = item['t']
            text += '& \\textit{' + f'{t:-.1f}' + '}\n'

        text += '\\\\ \\hline \n\n'

    text += '% ------ AUTO CODE END\n\n'

    log(text, d0, name, 'show')


def run_show_iter(d, name='calc1'):
    """Show results of the previous calculations for "iter"."""
    res = load(d, name, 'iter')
    if res is None:
        log('>>> Iter-result is not available', d, name, 'show')
        return

    text = '>>> Iter-result (png file with plot): \n\n'

    plt.figure(figsize=(16, 8))
    plt.xlabel('number of queries')
    plt.ylabel('absolute error')

    for i, func in enumerate(get_funcs(d), 1):
        v = [item['e'] for item in res[func.name]]
        plt.plot(EVALS_LIST, v, label=f'F{i}', marker='o')

    plt.grid()
    plt.semilogx()
    plt.semilogy()
    plt.legend(loc='best', ncol=5, fontsize=20)

    fpath = f'./demo_calc/res_plot/{name}_iter_{d}dim.png'
    plt.savefig(fpath, bbox_inches='tight')
    text += f'Figure saved to file "{fpath}"\n\n'

    log(text, d, name, 'show')


def run_show_quan(d, name='calc1'):
    """Show results of the previous calculations for "quan"."""
    res = load(d, name, 'quan')
    if res is None:
        log('>>> Quan-result is not available', d, name, 'show')
        return

    text = '>>> Quan-result (part of latex table): \n\n'

    text += '% ------ AUTO CODE START\n'

    for i, n in enumerate(N_LIST):
        text += '\\multirow{2}{*}{' + str(n) + '}'

        text += '\n& TT '
        for func in get_funcs(d):
            v = res[func.name]['q0'][i]['e']
            text += f'& {v:-8.1e} '
        text += '\\\\'

        text += '\n& QTT '
        for func in get_funcs(d):
            v = res[func.name]['q2'][i]['e']
            text += f'& {v:-8.1e} '
        text += ' \\\\ \\hline \n'

    text += '% ------ AUTO CODE END\n\n'

    log(text, d, name, 'show')


def run_show_rank(d, name='calc1'):
    """Show results of the previous calculations for "rank"."""
    res = load(d, name, 'rank')
    if res is None:
        log('>>> Rank-result is not available', d, name, 'show')
        return

    text = '>>> Rank-result (png file with plot): \n\n'

    plt.figure(figsize=(16, 8))
    plt.xlabel('rank')
    plt.ylabel('absolute error')
    plt.xticks(R_LIST)

    for i, func in enumerate(get_funcs(d), 1):
        v = [item['e'] for item in res[func.name]]
        plt.plot(R_LIST, v, label=f'F{i}', marker='o')

    plt.grid()
    plt.semilogy()
    plt.legend(loc='best', ncol=5, fontsize=20)

    fpath = f'./demo_calc/res_plot/{name}_rank_{d}dim.png'
    plt.savefig(fpath, bbox_inches='tight')
    text += f'Figure saved to file "{fpath}"\n\n'

    log(text, d, name, 'show')


def save(res, d, name, kind):
    fpath = f'./demo_calc/res_data/{name}_{kind}_{d}dim.pickle'
    with open(fpath, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def solve(opt, d, name, kind, reps=1, d_log=None):
    t, y, e, m = [], [], [], []
    for rep in range(reps):
        np.random.seed(rep)
        random.seed(rep)

        opt.init()
        opt.solve()

        log(opt.info(), d_log or d, name, kind)

        t.append(opt.t)
        y.append(opt.y)
        e.append(opt.e_y)
        m.append(opt.m)

    if reps > 1:
        print(f'--> Mean error / time : {np.mean(e):-7.1e} / {np.mean(t):.2f}')

    return {
        't': np.mean(t),
        'e': np.mean(e),
        'e_var': np.var(e),
        'e_min': np.min(e),
        'e_max': np.max(e),
        'e_avg': np.mean(e),
        'e_all': e,
        'y_all': y,
        'y_real': opt.y_real,
        'evals': int(np.mean(m))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Investigating TTOpt performance for analytic functions')
    parser.add_argument('-d', default=10,
        type=int, help='dimension d')
    parser.add_argument('-p', default=2,
        type=int, help='grid param p')
    parser.add_argument('-q', default=25,
        type=int, help='grid param q')
    parser.add_argument('-r', default=4,
        type=int, help='rank')
    parser.add_argument('--evals', default=1.E+5,
        type=float, help='computational budget')
    parser.add_argument('--reps', default=1,
        type=int, help='repetitions')
    parser.add_argument('--name', default='calc1',
        type=str, help='calculation name (the corresponding prefix will be used for the files with results)')
    parser.add_argument('--kind', default='comp',
        type=str, help='kind of calculation ("comp" - compare different solvers; "dims" - check dependency on dimension number; "iter" - check dependency on number of calls for target function; "quan" - check effect of qtt-usage; "rank" - check dependency on rank; "show" - show results of the previous calculations)')
    parser.add_argument('--verb', default=False,
        type=bool, help='if True, then intermediate results of the optimization process will be printed to the console')

    args = parser.parse_args()

    if args.kind == 'comp':
        run_comp(args.d, args.p, args.q, args.r, args.evals, args.reps,
            args.name, args.verb)
    elif args.kind == 'dims':
        run_dims(args.p, args.q, args.r, args.reps,
            args.name, args.verb)
    elif args.kind == 'iter':
        run_iter(args.d, args.p, args.q, args.r, args.reps,
            args.name, args.verb)
    elif args.kind == 'quan':
        run_quan(args.d, args.r, args.evals, args.reps,
            args.name, args.verb)
    elif args.kind == 'rank':
        run_rank(args.d, args.p, args.q, args.evals, args.reps,
            args.name, args.verb)
    elif args.kind == 'show':
        run_show(args.d, args.name)
    else:
        raise ValueError(f'Invalid kind of calculation "{args.kind}"')
