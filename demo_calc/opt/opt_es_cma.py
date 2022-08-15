import numpy as np
from time import perf_counter as tpc


from es import CMAES
from opt import Opt


class OptESCMA(Opt):
    """Minimizer based on the CMA-ES algorithm."""
    name = 'ES-CMA'

    def prep(self, popsize=255, iters=40000, sigma=0.1, decay=0.01, seed=42):
        self.popsize = int(popsize)
        self.iters = int(iters)
        self.sigma = sigma
        self.decay = decay
        self.seed =  int(seed)

        return self

    def solve(self):
        t = tpc()

        self.run_estool(CMAES(
            self.d,
            popsize=self.popsize,
            sigma_init=self.sigma,
            weight_decay=self.decay,
            seed=self.seed,
            x0=self.a + (self.b - self.a) * np.random.uniform()))

        self.t = tpc() - t

    def to_dict(self):
        res = super().to_dict()
        res['popsize'] = self.popsize
        res['iters'] = self.iters
        res['sigma'] = self.sigma
        res['decay'] = self.decay
        return res
