import numpy as np
from time import perf_counter as tpc


from .ttopt_raw import ttopt
from .ttopt_raw import ttopt_find


class TTOpt():
    """Multidimensional optimizer based on the cross-maximum-volume principle.

    Class for computation of the minimum or maximum for the implicitly given
    d-dimensional array (tensor) or a function of d-dimensional argument. An
    adaptive method based on the tensor train (TT) approximation and the
    cross-maximum volume principle is used. Cache of requested values (its
    usage leads to faster computation if one point is computed for a long time)
    and QTT-based representation of the grid (its usage in many cases leads to
    more accurate results) are supported.

    Args:
        f (function): the function of interest. Its argument X should represent
            several spatial points for calculation (is 2D numpy array of the
            shape [samples, d]) if "is_vect" flag is True, and it is one
            spatial point for calculation (is 1D numpy array of the shape [d])
            in the case if "is_vect" flag is False. For the case of the tensor
            approximation (if "is_func" flag is False), the argument X relates
            to the one or many (depending on the value of the flag "is_vect")
            multi-indices of the corresponding array/tensor. Function should
            return the values in the requested points (is 1D numpy array of the
            shape [samples] of float or only one float value depending on the
            value of "is_vect" flag). If "with_opt" flag is True, then function
            should also return the second argument (is 1D numpy array of the
            shape [samples] of any or just one value depending on the "is_vect"
            flag) which is the auxiliary quantity corresponding to the
            requested points (it is used for debugging and in specific parallel
            calculations; the value of this auxiliary quantity related to the
            "argmin / argmax" point will be passed to "callback" function).
        d (int): number of function dimensions.
        a (float or list of len d of float): grid lower bounds for every
            dimension. If a number is given, then this value will be used for
            each dimension.
        b (float or list of len d of float): grid upper bounds for every
            dimension. If a number is given, then this value will be used for
            each dimension.
        n (int or list of len d of int): number of grid points for every
            dimension. If a number is given, then this value will be used for
            each dimension. If this parameter is not specified, then instead of
            it the values for both "p" and "q" should be set.
        p (int): the grid size factor (if is given, then there will be n=p^q
            points for each dimension). This parameter can be specified instead
            of "n". If this parameter is specified, then the parameter "q" must
            also be specified, and in this case the QTT-based approach will be
            used.
        q (int): the grid size factor (if is given, then there will be n=p^q
            points for each dimension). This parameter can be specified instead
            of "n". If this parameter is specified, then the parameter "p" must
            also be specified, and in this case the QTT-based approach will be
            used.
        evals (int or float): the number of requests to the target function
            that will be made.
        name (str): optional display name for the function of interest. It is
            the empty string by default.
        callback (function): optional function that will be called after each
            optimization step (in Func.comp_opt) with related info (it is used
            for debugging and in specific parallel calculations).
        x_opt_real (list of len d): optional real value of x-minimum or maximum
            (x). If this value is specified, then it will be used to display the
            current approximation error within the algorithm iterations (this
            is convenient for debugging and testing/research).
        y_opt_real (float): optional real value of y-optima (y=f(x)). If
            this value is specified, then it will be used to display the
            current approximation error within the algorithm iterations (this
            is convenient for debugging and testing/research).
        is_func (bool): if flag is True, then we optimize the function (the
            arguments of f correspond to continuous spatial points), otherwise
            we approximate the tensor (the arguments of f correspond to
            discrete multidimensional tensor multi-indices). It is True by
            default.
        is_vect (bool): if flag is True, then function should accept 2D
            numpy array of the shape [samples, d] (batch of points or indices)
            and return 1D numpy array of the shape [samples]. Otherwise, the
            function should accept 1D numpy array (one multidimensional point)
            and return the float value. It is True by default.
        with_cache (bool): if flag is True, then all requested values are
            stored and retrieved from the storage upon repeated requests.
            Note that this leads to faster computation if one point is
            computed for a long time. On the other hand, this can in some
            cases slow down the process, due to the additional time spent
            on checking the storage and using unvectorized slow loops in
            python. It is False by default.
        with_log (bool): if flag is True, then text messages will be
            displayed during the optimizer query process. It is False by
            default.
        with_opt (bool): if flag is True, then function of interest returns
            opts related to output y (scalar or vector) as second argument
            (it will be also saved and passed to "callback" function). It is
            False by default.
        with_full_info (bool): if flag is True, then the full information will
            be saved, including multi-indices of requested points (it is used
            by animation function) and best found multi-indices and points.
            Note that the inclusion of this flag can significantly slow down
            the process of the algorithm. It is False by default.
        with_wrn (bool): if flag is True, then warning messages will be
            presented (in the current version, it can only be messages about
            early convergence when using the cache). It is True by default.

    Note:
        Call "calc" to evaluate function for one tensor multi-index and call
        "comp" to evaluate function in the set of multi-indices (both of these
        functions can be called regardless of the value of the flag "is_vect").
        Call "minimize" / "maximize" to find the global minimum / maximum of
        the function of interest by the TTOpt-algorithm.

    """

    def __init__(self, f, d, a=None, b=None, n=None, p=None, q=None,
                 evals=None, name=None, callback=None, x_opt_real=None,
                 y_opt_real=None, is_func=True, is_vect=True, with_cache=False,
                 with_log=False, with_opt=False, with_full_info=False,
                 with_wrn=True):
        # Set the target function and its dimension:
        self.f = f
        self.d = int(d)

        # Set grid lower bound:
        if isinstance(a, (int, float)):
            self.a = np.ones(self.d, dtype=float) * a
        elif a is not None:
            self.a = np.asanyarray(a, dtype=float)
        else:
            if is_func:
                raise ValueError('Grid lower bound (a) should be set')
            self.a = None
        if self.a is not None and self.a.size != self.d:
            raise ValueError('Grid lower bound (a) has invalid shape')

        # Set grid upper bound:
        if isinstance(b, (int, float)):
            self.b = np.ones(self.d, dtype=float) * b
        elif b is not None:
            self.b = np.asanyarray(b, dtype=float)
        else:
            if is_func:
                raise ValueError('Grid upper bound (b) should be set')
            self.b = None
        if self.b is not None and self.b.size != self.d:
            raise ValueError('Grid upper bound (b) has invalid shape')

        # Set number of grid points:
        if n is None:
            if p is None or q is None:
                raise ValueError('If n is not set, then p and q should be set')
            self.p = int(p)
            self.q = int(q)
            self.n = np.ones(self.d * self.q, dtype=int) * self.p
            self.n_func = np.ones(self.d, dtype=int) * (self.p**self.q)
        else:
            if p is not None or q is not None:
                raise ValueError('If n is set, then p and q should be None')
            self.p = None
            self.q = None
            if isinstance(n, (int, float)):
                self.n = np.ones(self.d, dtype=int) * int(n)
            else:
                self.n = np.asanyarray(n, dtype=int)
            self.n_func = self.n.copy()
        if self.n_func.size != self.d:
            raise ValueError('Grid size (n/p/q) has invalid shape')

        # Set other options according to the input arguments:
        self.evals = int(evals) if evals else None
        self.name = name or ''
        self.callback = callback
        self.x_opt_real = x_opt_real
        self.y_opt_real = y_opt_real
        self.is_func = bool(is_func)
        self.is_vect = bool(is_vect)
        self.with_cache = bool(with_cache)
        self.with_log = bool(with_log)
        self.with_opt = bool(with_opt)
        self.with_full_info = bool(with_full_info)
        self.with_wrn = bool(with_wrn)

        # Inner variables:
        self.cache = {}     # Cache for the results of requests to function
        self.cache_opt = {} # Cache for the options while requests to function
        self.k_cache = 0    # Number of requests, then cache was used
        self.k_cache_curr = 0
        self.k_evals = 0    # Number of requests, then function was called
        self.k_evals_curr = 0
        self.t_evals = 0.   # Total time of function calls
        self.t_total = 0.   # Total time of computations (including cache usage)
        self.t_minim = 0    # Total time of work for minimizator
        self._opt = None    # Function opts related to its output

        # Current optimum:
        self.i_opt = None
        self.x_opt = None

        # Approximations for argopt/opt/opts of the function while iterations:
        self.I_list = []
        self.i_opt_list = []
        self.x_opt_list = []
        self.y_opt_list = []
        self.opt_opt_list = []
        self.evals_opt_list = []
        self.cache_opt_list = []

    @property
    def e_x(self):
        """Current error for approximation of arg-opt of the function."""
        if self.x_opt_real is not None and self.x_opt is not None:
            return np.linalg.norm(self.x_opt - self.x_opt_real)

    @property
    def e_y(self):
        """Current error for approximation of the optimum of the function."""
        if self.y_opt_real is not None and self.y_opt is not None:
            return np.abs(self.y_opt - self.y_opt_real)

    @property
    def k_total(self):
        """Total number of requests (both function calls and cache usage)."""
        return self.k_cache + self.k_evals

    @property
    def opt_opt(self):
        """Current value of option of the function related to opt-point."""
        return self.opt_opt_list[-1] if len(self.opt_opt_list) else None

    @property
    def t_evals_mean(self):
        """Average time spent to real function call for 1 point."""
        return self.t_evals / self.k_evals if self.k_evals else 0.

    @property
    def t_total_mean(self):
        """Average time spent to return one function value."""
        return self.t_total / self.k_total if self.k_total else 0.

    @property
    def y_opt(self):
        """Current approximation of optimum of the function of interest."""
        return self.y_opt_list[-1] if len(self.y_opt_list) else None

    def calc(self, i):
        """Calculate the function for the given multiindex.

        Args:
            i (np.ndarray): the input for the function, that is 1D numpy array
                of the shape [d] of int (indices).

        Returns:
            float: the output of the function.

        """
        self.k_cache_curr = 0

        if self.is_vect:
            return self.comp(i.reshape(1, -1))[0]

        t_total = tpc()

        if not self.with_cache:
            y = self._eval(i)
            self.t_total += tpc() - t_total
            return y

        s = self.i2s(i.astype(int).tolist())

        if not s in self.cache:
            y = self._eval(i)
            if y is None:
                return y
            self.cache[s] = y
            self.cache_opt[s] = self._opt
        else:
            y = self.cache[s]
            self._opt = self.cache_opt[s]
            self.k_cache_curr = 1
            self.k_cache += self.k_cache_curr

        self.t_total += tpc() - t_total

        return y

    def comp(self, I):
        """Compute the function for the set of multi-indices (samples).

        Args:
            I (np.ndarray): the inputs for the function, that are collected in
                2D numpy array of the shape [samples, d] of int (indices).

        Returns:
            np.ndarray: the outputs of the function, that are collected in
            1D numpy array of the shape [samples].

        Note:
            The set of points (I) should not contain duplicate points. If it
            contains duplicate points (that are not in the cache), then each of
            them will be recalculated without using the cache.

        """
        self.k_cache_curr = 0

        if not self.is_vect:
            Y, _opt = [], []
            for i in I:
                y = self.calc(i)
                if y is None:
                    return None
                Y.append(y)
                _opt.append(self._opt)
            self._opt = _opt
            return np.array(Y)

        t_total = tpc()

        if not self.with_cache:
            Y = self._eval(I)
            self.t_total += tpc() - t_total
            return Y

        # Requested points:
        I = I.tolist()

        # Points that are not presented in the cache:
        J = [i for i in I if self.i2s(i) not in self.cache]
        self.k_cache_curr = len(I) - len(J)
        self.k_cache += self.k_cache_curr

        # We add new points (J) to the storage:
        if len(J):
            Z = self._eval(J)
            if Z is None:
                return None
            for k, j in enumerate(J):
                s = self.i2s(j)
                self.cache[s] = Z[k]
                self.cache_opt[s] = self._opt[k]

        # We obtain the values for requested points from the updated storage:
        Y = np.array([self.cache[self.i2s(i)] for i in I])
        self._opt = np.array([self.cache_opt[self.i2s(i)] for i in I])

        self.t_total += tpc() - t_total

        return Y

    def comp_opt(self, I, i_opt=None, y_opt=None, opt_opt=None):
        """Compute the function for the set of points and save current optimum.

        This helper function (this is wrapper for function "comp") can be
        passed to the optimizer. When making requests, the optimizer must pass
        the grid points of interest (I) as arguments, as well as the current
        approximation of the argmin / argmax (i_opt), the corresponding value
        (y_opt) and related option value (opt_opt).

        """
        # We return None if the limit for function requests is exceeded:
        if self.evals is not None and self.k_evals >= self.evals:
            return None, None

        # We return None if the number of requests to the cache is 2 times
        # higher than the number of requests to the function:
        if self.with_cache:
            if self.k_cache >= self.evals and self.k_cache >= 2 * self.k_evals:
                text = '!!! TTOpt warning : '
                text += 'the number of requests to the cache is 2 times higher '
                text += 'than the number of requests to the function. '
                text += 'The work is finished before max func-evals reached.'
                if self.with_wrn:
                    print(text)
                return None, None

        # We truncate the list of requested points if it exceeds the limit:
        eval_curr = I.shape[0]
        is_last = self.evals is not None and self.k_evals+eval_curr>=self.evals
        if is_last:
            I = I[:(self.evals-self.k_evals), :]

        if self.q:
            # The QTT is used, hence we should transform the indices:
            if I is not None:
                I = self.qtt_parse_many(I)
            if i_opt is not None:
                i_opt = self.qtt_parse_many(i_opt.reshape(1, -1))[0, :]

        Y = self.comp(I)

        # If this is last iteration, we should "manually" check for y_opt_new:
        if is_last:
            i_opt, y_opt, opt_opt = ttopt_find(
                I, Y, self._opt, i_opt, y_opt, opt_opt, self.is_max)

        if i_opt is None:
            return Y, self._opt

        if self.is_func:
            x_opt = self.i2x(i_opt)
        else:
            x_opt = i_opt.copy()

        self.i_opt = i_opt.copy()
        self.x_opt = x_opt.copy()

        self.y_opt_list.append(y_opt)
        self.opt_opt_list.append(opt_opt)
        self.evals_opt_list.append(self.k_evals_curr)
        self.cache_opt_list.append(self.k_cache_curr)

        if self.with_full_info:
            self.I_list.append(I)
            self.i_opt_list.append(self.i_opt.copy())
            self.x_opt_list.append(self.x_opt.copy())

        if self.is_max:
            is_better = len(self.y_opt_list)==1 or (y_opt > self.y_opt_list[-2])
        else:
            is_better = len(self.y_opt_list)==1 or (y_opt < self.y_opt_list[-2])

        if self.callback and is_better:
            last = {'last': [x_opt, y_opt, i_opt, opt_opt, self.k_evals]}
            self.callback(last)

        if self.with_log:
            print(self.info(is_final=False))

        return Y, self._opt

    def i2s(self, i):
        """Transform array of int like [1, 2, 3] into string like '1-2-3'."""
        return '-'.join([str(v) for v in i])

    def i2x(self, i):
        """Transform multiindex into point of the uniform grid."""
        t = i * 1. / (self.n_func - 1)
        x = t * (self.b - self.a) + self.a
        return x

    def i2x_many(self, I):
        """Transform multiindices (samples) into grid points."""
        A = np.repeat(self.a.reshape((1, -1)), I.shape[0], axis=0)
        B = np.repeat(self.b.reshape((1, -1)), I.shape[0], axis=0)
        N = np.repeat(self.n_func.reshape((1, -1)), I.shape[0], axis=0)
        T = I * 1. / (N - 1)
        X = T * (B - A) + A
        return X

    def info(self, with_e_x=True, with_e_y=True, is_final=True):
        """Return text description of the progress of optimizer work."""
        text = ''

        if self.name:
            name = self.name + f'-{self.d}d'
            name += ' ' * max(0, 10 - len(name))
            text += name + ' | '

        if self.with_cache:
            text += f'evals={self.k_evals:-8.2e}+{self.k_cache:-8.2e} | '
        else:
            text += f'evals={self.k_total:-8.2e} | '

        if is_final:
            text += f't_all={self.t_minim:-8.2e} | '
        else:
            text += f't_cur={self.t_total:-8.2e} | '

        if self.y_opt_real is None and self.y_opt is not None:
            text += f'y={self.y_opt:-13.6e} '
        else:
            if with_e_x and self.e_x is not None:
                text += f'e_x={self.e_x:-8.2e} '
            if with_e_y and self.e_y is not None:
                text += f'e_y={self.e_y:-8.2e} '

        return text

    def optimize(self, rank=4, Y0=None, seed=42, fs_opt=1., is_max=False,
                 add_opt_inner=True, add_opt_outer=False, add_opt_rect=False,
                 add_rnd_inner=False, add_rnd_outer=False, J0=None):
        """Perform the function optimization process by TT-based approach.

        Args:
            rank (int): maximum TT-rank.
            Y0 (list of 3D np.ndarrays of float): optional initial tensor in
                the TT format as a list of the TT-cores.
            seed (int): random seed for the algorithm initialization. It is
                used only if Y0 and J0 are not set.
            fs_opt (float): the parameter of the smoothing function. If it is
                None, then "arctan" function will be used. Otherwise, the
                function "exp(-1 * fs_opt * (p - p0))" will be used.
            is_max (bool): if flag is True, then maximization will be performed.

        """
        t_minim = tpc()
        self.is_max = is_max

        i_opt, y_opt = ttopt(self.comp_opt, self.n, rank, None, Y0, seed,
                fs_opt, add_opt_inner, add_opt_outer, add_opt_rect,
                add_rnd_inner, add_rnd_outer, J0, is_max)

        self.t_minim = tpc() - t_minim

    def qtt_parse_many(self, I_qtt):
        """Transform tensor indices from QTT (long) to base (short) format."""
        samples = I_qtt.shape[0]
        n_qtt = [self.n[0]]*self.q
        I = np.zeros((samples, self.d))
        for i in range(self.d):
            J_curr = I_qtt[:, self.q*i:self.q*(i+1)].T
            I[:, i] = np.ravel_multi_index(J_curr, n_qtt, order='F')
        return I

    def s2i(self, s):
        """Transforms string like '1-2-3' into array of int like [1, 2, 3]."""
        return np.array([int(v) for v in s.split('-')], dtype=int)

    def _eval(self, i):
        """Helper that computes target function in one or many points."""
        t_evals = tpc()

        i = np.asanyarray(i, dtype=int)
        is_many = len(i.shape) == 2

        if self.is_func:
            x = self.i2x_many(i) if is_many else self.i2x(i)
        else:
            x = i

        if self.with_opt:
            y, self._opt = self.f(x)
            if y is None:
                return None
        else:
            y = self.f(x)
            if y is None:
                return None
            self._opt = [None for _ in range(y.size)] if is_many else None

        self.k_evals_curr = y.size if is_many else 1
        self.k_evals += self.k_evals_curr
        self.t_evals += tpc() - t_evals

        return y
