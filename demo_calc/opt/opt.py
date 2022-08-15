import numpy as np


class Opt():
    """Base class for minimizer.

    Note:
        Concrete minimizers should extend this class.

    """
    name = 'Base minimizer'

    def __init__(self, func, d, a, b, x_min=None, y_min=None, verb=True):
        self._f0 = func          # Scalar function
        self._f = func           # Vector function
        self.d = d               # Dimension
        self.a = a               # Grid lower limit
        self.b = b               # Grid upper limit
        self.x_real = x_min      # Real x (arg) for minimum (for error check)
        self.y_real = y_min      # Real y (f(x)) for minimum (for error check)
        self.verb = verb         # Verbosity of output (True/False)

        self.prep()
        self.init()

    @property
    def e_x(self):
        if self.x_real is None or self.x is None:
            return None
        return np.linalg.norm(self.x_real - self.x)

    @property
    def e_y(self):
        if self.y_real is None or self.y is None:
            return None
        return abs(self.y_real - self.y)

    def info(self, text_spec=''):
        text = ''
        text += f'{self.name}' + ' '*(12 - len(self.name)) + ' | '

        text += f't={self.t:-6.1f}'

        if self.e_y is not None:
            text += f' | ey={self.e_y:-7.1e}'

        text += f' | evals={self.m:-7.1e}'

        if text_spec:
            text += f' | {text_spec}'

        return text

    def init(self):
        self.t = 0.              # Work time (sec)
        self.m = 0               # Number of function calls
        self.x = None            # Found x (arg) for minimum
        self.y = None            # Found y (f(x)) for minimum

        return self

    def f0(self, x):
        self.m += 1
        return self._f0(x)

    def f0_max(self, x):
        return -self.f0(x)

    def f(self, X):
        self.m += X.shape[0]
        return self._f(X)

    def f_max(self, X):
        return -self.f(X)

    def prep(self):
        return self

    def run_estool(self, solver):
        for j in range(self.iters):
            solutions = solver.ask()

            fitness_list = np.zeros(solver.popsize)
            for i in range(solver.popsize):
                fitness_list[i] = self.f_max(solutions[i].reshape(1, -1))[0]

            solver.tell(fitness_list)
            result = solver.result()

            self.x = result[0]
            self.y = result[1]

            if self.verb and (j+1) % 10 == 0:
                text = ''
                text += f'k={self.m:-8.2e} | '
                text += f'iter={j+1:-6d} | '
                text += f'e_y={self.e_y:-8.2e} '
                print(text)

    def solve(self):
        raise NotImplementedError()

    def to_dict(self):
        return {
            'name': self.name,
            'd': self.d,
            'a': self.a,
            'b': self.b,
            't': self.t,
            'm': self.m,
            'y': self.y,
            'y_real': self.y_real,
            'e_x': self.e_x,
            'e_y': self.e_y,
        }
