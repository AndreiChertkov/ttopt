import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_init


np.random.seed(16333)


def run(d=10, p=2, q=25, evals=1.E+5, rmax=4, with_cache=False):
    n = np.ones(d * q, dtype=int) * p
    Y0 = ttopt_init(n, rmax)

    for func in teneva.func_demo_all(d=d):
        name = func.name + ' ' * (15 - len(func.name))
        tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
            name=name, x_min_real=func.x_min, y_min_real=func.y_min,
            p=p, q=q, evals=evals, with_cache=with_cache)
        tto.minimize(rmax, Y0, fs_opt=1.)
        print(tto.info(with_e_x=False))


def run_many(d=10, p=2, q=25, evals=1.E+5, rmax=4):
    for func in teneva.func_demo_all(d=d):
        lim = func.b[0] - func.a[0]
        for fs_opt in [None, 1000., 100., 10., 1., 0.1, 0.01]:
            for i in range(5):
                name = func.name + ' ' * (15 - len(func.name))
                tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                    name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                    p=p, q=q, evals=evals)
                tto.minimize(rmax=rmax, fs_opt=fs_opt)
                print(tto.info(with_e_x=False) + f' | {lim:-6.2f} | opt = {fs_opt}')
            print('')
        print('\n')


def run_one(d=10, p=2, q=25, evals=1.E+7, rmax=4, with_cache=False):
    n = np.ones(d * q, dtype=int) * p
    Y0 = ttopt_init(n, rmax)

    func = teneva.func_demo_all(d=d, names=['Rosenbrock'])[0]
    tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
        name=func.name, x_min_real=func.x_min, y_min_real=func.y_min,
        p=p, q=q, evals=evals, with_log=True, with_cache=with_cache)
    tto.minimize(rmax, Y0, fs_opt=1.)
    print(tto.info(with_e_x=False))


def run_rep(d=10, p=2, q=25, evals=1.E+5, rmax=4, reps=10):
    for func in teneva.func_demo_all(d=d):
        e_list = []
        t_list = []
        for i in range(reps):
            name = func.name + ' ' * (15 - len(func.name))
            tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                p=p, q=q, evals=evals)
            tto.minimize(rmax=rmax)
            e_list.append(tto.e_y)
            t_list.append(tto.t_minim)

        text = f'{name} | '
        text += 'err (avg/min/max): '
        text += f'{np.mean(e_list):-7.1e} / '
        text += f'{ np.min(e_list):-7.1e} / '
        text += f'{ np.max(e_list):-7.1e} | '
        text += f't: {np.mean(t_list):-7.4f}'
        print(text)


if __name__ == '__main__':
    run()
