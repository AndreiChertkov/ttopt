import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_init


np.random.seed(16333)


def run(d=10, n=1024, evals=1.E+6, rmax=4, reps=10, fs_opt=1., add_opt_inner=True, add_opt_outer=False, add_opt_rect=False, add_rnd_inner=False, add_rnd_outer=False, num=1):
    errors, times = [], []
    for func in teneva.func_demo_all(d=d):
        if func.name == 'Michalewicz':
            continue
        e_list = []
        t_list = []
        for i in range(reps):
            name = func.name + ' ' * (15 - len(func.name))

            tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                n=n, evals=evals)
            tto.minimize(rmax, None, fs_opt, add_opt_inner, add_opt_outer,
                add_opt_rect, add_rnd_inner, add_rnd_outer)
            e_list.append(tto.e_y)
            t_list.append(tto.t_minim)

        errors.append(np.mean(e_list))
        times.append(np.mean(t_list))

    print(f'#{num:-2d} | e : ', ' | '.join([f'{e:-7.1e}' for e in errors]))


if __name__ == '__main__':
    run(num=1, add_opt_inner=False, add_opt_outer=False, add_opt_rect=False,
        add_rnd_inner=False, add_rnd_outer=False)
    run(num=2, add_opt_inner=False, add_opt_outer=False, add_opt_rect=True,
        add_rnd_inner=False, add_rnd_outer=False)
    run(num=3, add_opt_inner=True, add_opt_outer=False, add_opt_rect=False,
        add_rnd_inner=False, add_rnd_outer=False)
    run(num=4, add_opt_inner=False, add_opt_outer=True, add_opt_rect=False,
        add_rnd_inner=False, add_rnd_outer=False)
    run(num=5, add_opt_inner=True, add_opt_outer=True, add_opt_rect=False,
        add_rnd_inner=False, add_rnd_outer=False)
    run(num=6, add_opt_inner=False, add_opt_outer=False, add_opt_rect=False,
        add_rnd_inner=True, add_rnd_outer=True)
    run(num=7, add_opt_inner=True, add_opt_outer=True, add_opt_rect=True,
        add_rnd_inner=True, add_rnd_outer=True)
