from time import perf_counter as tpc
import nevergrad as ng


from opt import Opt


class OptNB(Opt):
    """Minimizer based on the NoisyBandit method from "nevergrad" package."""
    name = 'NB'

    def prep(self, evals=1.E+7):
        self.evals = int(evals)

        return self

    def solve(self):
        t = tpc()

        par = ng.p.Array(shape=(self.d,), lower=self.a, upper=self.b)
        opt = ng.optimizers.registry['NoisyBandit'](budget=self.evals,
            parametrization=par, num_workers=1)
        self.x = opt.minimize(self.f0).value
        self.y = self.f0(self.x)

        self.t = tpc() - t


    def to_dict(self):
        res = super().to_dict()
        res['evals'] = self.evals
        return res
