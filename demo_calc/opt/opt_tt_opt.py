from time import perf_counter as tpc
from ttopt import TTOpt


from opt import Opt


class OptTTOpt(Opt):
    """Minimizer based on the TTOpt."""
    name = 'TTOpt'

    def info(self):
        if self.n is not None:
            text = f'n={self.n:-5d}'
        else:
            text = f'p={self.p:-1d}, q={self.q:-2d}'
        text += f', r={self.r:-3d}'
        return super().info(text)

    def prep(self, n=None, p=2, q=12, r=6, evals=1.E+7):
        self.n = n
        self.p = p
        self.q = q
        self.r = r
        self.evals = int(evals)

        return self

    def solve(self):
        t = tpc()

        ttopt = TTOpt(
            self.f,
            d=self.d,
            a=self.a,
            b=self.b,
            n=self.n,
            p=self.p,
            q=self.q,
            evals=self.evals,
            x_min_real=self.x_real,
            y_min_real=self.y_real,
            with_log=self.verb)
        ttopt.minimize(self.r)

        self.t = tpc() - t

        self.x = ttopt.x_min
        self.y = ttopt.y_min

    def to_dict(self):
        res = super().to_dict()
        res['n'] = self.n
        res['p'] = self.p
        res['q'] = self.q
        res['r'] = self.r
        res['evals'] = self.evals
        return res
