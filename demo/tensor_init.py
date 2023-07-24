"""The demo of using ttopt. Example for tensor minimization (special case).

We'll find the minimum for the given simple d-dimensional tensor with elements:
Y[i_1, i_2, ..., i_d] = (i_1 - 2)^2 + (i_2 - 3)^2 + i_2^4 + i_3^4 + ... + i_d^4.

We will use special method of initialization. Instead of a random tensor, we
manually construct a set of starting multi-indices for the search.

Run it from the root of the project as "python demo/tensor_init_spec.py".

As a result of the script work we expect the output in console like this:
"
...
Tensor-10d | evals=1.00e+05 | t_cur=8.92e-03 | e_x=0.00e+00 e_y=0.00e+00
----------------------------------------------------------------------
Tensor-10d | evals=1.00e+05 | t_all=1.58e-01 | e_x=0.00e+00 e_y=0.00e+00
y_opt :  0
i_opt :  [2 3 0 0 0 0 0 0 0 0]
"

"""
import numpy as np


from ttopt import TTOpt
from ttopt import ttopt_init


np.random.seed(42)


d = 10                      # Number of function dimensions
p = 2
q = 10
n = p**q                    # Mode size for the tensor
rank = 2                    # Maximum TT-rank while cross-like iterations
def f(I):                   # Target function (return tensor element)
    return (I[:, 0] - 2)**2 + (I[:, 1] - 3)**2 + np.sum(I[:, 2:]**4, axis=1)


# Real value of x-minima:
x_min_real = np.zeros(d)
x_min_real[0] = 2
x_min_real[1] = 3


# We initialize the TTOpt class instance with the correct parameters:
tto = TTOpt(
    f=f,                    # Function for minimization. X is [samples, dim]
    d=d,                    # Number of function dimensions
    n=n,                    # Number of grid points (number or list of len d)
    evals=1.E+5,            # Number of function evaluations
    name='Tensor',          # Function name for log (this is optional)
    x_opt_real=x_min_real,  # Real value of x-minima (x; this is for test)
    y_opt_real=0.,          # Real value of y-minima (y=f(x); this is for test)
    is_func=False,          # We approximate the tensor (not a function)
    with_log=True)


# We manually construct a set of starting multi-indices for the search (note
# that the list contains (d+1) items, the first and last items should be None):
J0 = [None for _ in range(d+1)]
J0[1] = np.zeros((rank, 1), dtype=int)
for k in range(1, d-1):
    ir = np.ones((rank, 1), dtype=int)
    ir *= 1 if k % 2 == 1 else 0
    J0[k+1] = np.hstack((J0[k], ir))


# And now we launching the minimizer:
tto.optimize(rank, J0=J0)


# We can extract the results of the computation:
i = tto.i_opt          # The found value of the minimum (multi-index)
y = tto.y_opt          # The found value of the minimum of the function (y=f(x))
k_c = tto.k_cache      # Total number of cache usage (should be 0 in this demo)
k_e = tto.k_evals      # Total number of requests to func (is always = evals)
k_t = tto.k_total      # Total number of requests (k_cache + k_evals)
t_f = tto.t_evals_mean # Average time spent to real function call for 1 point
                       # ... (see "ttopt.py" and docs for more details)


# We log the final state:
print('-' * 70 + '\n' + tto.info())
print('y_opt : ', y)
print('i_opt : ', i)
