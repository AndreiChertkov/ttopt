import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_init


mpl.rc('animation', html='jshtml')
mpl.rcParams['animation.embed_limit'] = 2**128


def animate(func, tto, frames=None, fpath=None):
    n = [tto.p**tto.q] * 2 if tto.p is not None else tto.n
    X1 = np.linspace(func.a[0], func.b[0], n[0])
    X2 = np.linspace(func.a[1], func.b[1], n[1])
    X1, X2 = np.meshgrid(X1, X2)
    X = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    Y = func.get_f_poi(X)
    Y = Y.reshape(X1.shape)

    i_min = np.array((func.x_min - tto.a) / (tto.b - tto.a) * (tto.n_func - 1), dtype=int)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    title = func.name + ' function'
    title += f' | y_min={tto.y_min_real}'
    title += f' | n = {tto.n}' if tto.p is None else f' | n = {tto.p}^{tto.q}'
    ax1.set_title(title, fontsize=16)

    surf = ax1.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, ax=ax1, shrink=0.3, aspect=10)

    ax2.imshow(Y, cmap=cm.coolwarm, alpha=0.8)
    ax2.scatter(i_min[0], i_min[1], s=300, c='#ffbf00', marker='*', alpha=0.8)
    ax2.set_xlim(0, n[0])
    ax2.set_ylim(0, n[1])

    img_min = ax2.scatter(0, 0, s=150, c='#EE17DA', marker='D')
    img_req = ax2.scatter(0, 0, s= 70, c='#8b1d1d')
    # img_req_old = ax2.scatter(0, 0, s= 50, c='#8b1d1d', alpha=0.2)
    img_hist, = ax2.plot([], [], '--', c='#485536', linewidth=1, markersize=0)

    def update(k, *args):
        x = tto.x_min_list[k]
        i = np.array((x - tto.a) / (tto.b - tto.a) * (tto.n_func - 1), dtype=int)
        img_min.set_offsets(np.array([i[0], i[1]]))

        I = tto.I_list[k]
        img_req.set_offsets(I)

        #if k > 0:
        #    I = tto.I_list[k-1]
        #    img_req_old.set_offsets(I)

        pois_x, pois_y = [], []
        for x in tto.x_min_list[:(k+1)]:
            i = (x - tto.a) / (tto.b - tto.a) * (tto.n_func - 1)
            pois_x.append(i[0])
            pois_y.append(i[1])
        img_hist.set_data(pois_x, pois_y)

        m = sum(tto.evals_min_list[:(k+1)])
        y = tto.y_min_list[k]
        e = np.abs(y - tto.y_min_real)
        ax2.set_title(f'Queries: {m:-7.1e} | Error : {e:-7.1e}', fontsize=20)

        return img_min, img_req, img_hist

    frames = frames or len(tto.x_min_list)
    anim = animation.FuncAnimation(fig, update, interval=30,
        frames=frames, blit=True, repeat=False)

    if fpath:
        anim.save(fpath, writer='pillow', fps=0.7)

    return anim


def run(d=2, p=2, q=12, evals=1.E+4, rmax=4, with_cache=False):
    n = np.ones(d * q, dtype=int) * p
    Y0 = ttopt_init(n, rmax)

    for func in teneva.func_demo_all(d=d):
        name = func.name + ' ' * (15 - len(func.name))
        tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
            name=name, x_min_real=func.x_min, y_min_real=func.y_min,
            p=p, q=q, evals=evals, with_cache=with_cache, with_full_info=True)
        tto.minimize(rmax, Y0, fs_opt=1.)
        print(tto.info(with_e_x=False))

        fpath = f'./check/animation/{func.name}.gif'
        animate(func, tto, fpath=fpath)


def run_base(d=2, n=256, evals=1.E+4, rmax=4, with_cache=False):
    n = [n] * d
    Y0 = ttopt_init(n, rmax)

    for func in teneva.func_demo_all(d=d):
        name = func.name + ' ' * (15 - len(func.name))
        tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
            name=name, x_min_real=func.x_min, y_min_real=func.y_min,
            n=n, evals=evals, with_cache=with_cache, with_full_info=True)
        tto.minimize(rmax, Y0, fs_opt=1.)
        print(tto.info(with_e_x=False))

        fpath = f'./check/animation/{func.name}_base.gif'
        animate(func, tto, fpath=fpath)


if __name__ == '__main__':
    np.random.seed(16333)
    run()
    run_base()
