import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_init


np.random.seed(16333)


def run(d=10, p=2, q=25, evals=1.E+5, rmax=4, reps=10):
    text = ''
    text += '-'*76 + '\n'
    text += 'Function     | Old        | New        | New + cache | '
    text += 'Time (old/new/new+c.)' + '\n'
    text += '-'*76
    print(text)

    for func in teneva.func_demo_all(d=d):
        e_list_old, e_list, e_list_cache = [], [], []
        t_list_old, t_list, t_list_cache = [], [], []

        for i in range(reps):
            name = func.name + ' ' * (12 - len(func.name))
            tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                p=p, q=q, evals=evals, use_old=True)
            tto.minimize(rmax=rmax, fs_opt=None)
            e_list_old.append(tto.e_y)
            t_list_old.append(tto.t_minim)

            tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                p=p, q=q, evals=evals)
            tto.minimize(rmax=rmax)
            e_list.append(tto.e_y)
            t_list.append(tto.t_minim)

            tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                p=p, q=q, evals=evals, with_cache=True, with_wrn=False)
            tto.minimize(rmax=rmax)
            e_list_cache.append(tto.e_y)
            t_list_cache.append(tto.t_minim)

        e_old, t_old = np.mean(e_list_old), np.mean(t_list_old)
        e, t = np.mean(e_list), np.mean(t_list)
        e_cache, t_cache = np.mean(e_list_cache), np.mean(t_list_cache)

        text = f'{name} | '
        text += f'{e_old:-7.1e}    | {e:-7.1e}    | {e_cache:-7.1e}     |'
        text += f'{t_old:-5.1f} / {t:-5.1f} / {t_cache:-5.1f}'
        print(text)

    print('-'*76)


if __name__ == '__main__':
    run()
