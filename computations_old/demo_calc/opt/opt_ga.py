from time import perf_counter as tpc


from es import SimpleGA
from opt import Opt


class OptGA(Opt):
    """Minimizer based on the Genetic Algorithm."""
    name = 'GA'

    def prep(self, popsize=256, iters=40000, sigma=0.1, decay=0.01):
        self.popsize = int(popsize)
        self.iters = int(iters)
        self.sigma = sigma
        self.decay = decay

        return self

    def solve(self):
        t = tpc()

        self.run_estool(SimpleGA(
            self.d,
            popsize=self.popsize,
            sigma_init=self.sigma,
            weight_decay=self.decay))

        self.t = tpc() - t

    def to_dict(self):
        res = super().to_dict()
        res['popsize'] = self.popsize
        res['iters'] = self.iters
        res['sigma'] = self.sigma
        res['decay'] = self.decay
        return res
