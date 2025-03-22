from functools import partial
import numpy as np

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(1,))
def LALG_SU_N(coef, N=3):
    """Takes as input a float array `coef[..., N^2-1]` and returns complex array `SU[..., N, N]`.
    The N^2-1 values in `coef` correspond to the coefficients of su(N) generators (e.g. N=3 corresponds to the Gell-Mann matrices divided by 2)\n
    Lie Algebra to Lie Group Special Unitary N."""
    
    su = LA_SU_N(coef, N)
    SU = jax.scipy.linalg.expm(1j*su)

    return SU

@partial(jax.jit, static_argnums=(1,))
def LA_SU_N(coef, N=3):
    """Takes as input a float (or real-complex) array `coef[..., N^2-1]` and returns complex array `su[..., N, N]` such that `expm(1j*su)` belongs to SU(N)."""
    coef = coef if jnp.iscomplexobj(coef) else jax.lax.complex(coef, jnp.zeros_like(coef))
    generators = _make_generators(N)
    
    su = jnp.einsum("...N,Nij->...ij", coef, generators)

    return su

@jax.jit
def proj_SU3(arr):
    D, V = jnp.linalg.eigh(arr.conj().mT @ arr)
    result = arr @ ((V * (jnp.expand_dims(D, axis=-2)**-0.5)) @ V.conj().mT)
    result = result * (jnp.expand_dims(jnp.linalg.det(result), axis=[-1, -2]) ** (-1/3))
    return result

@partial(jax.jit, static_argnums=(1,))
def unitary_violation(arr, aggregate="mean"):
    """Measure of how much `arr` violates unitarity. Computes |U^H @ U - I|, where || is the Frobenius norm.\n
    `aggregate`: `None, "mean", "max", "rms"`
    """

    I = np.broadcast_to(np.eye(arr.shape[-1], dtype=arr.dtype), arr.shape)

    violation = jnp.linalg.matrix_norm(arr.conj().mT @ arr - I, ord="fro")

    aggregate_fn = {
        None: lambda x: x,
        "mean": jnp.mean,
        "max": jnp.max,
        "rms": lambda x: jnp.sqrt(jnp.mean(jnp.square(x)))
    }
    
    return aggregate_fn[aggregate](violation)

def special_unitary_grad(func):
    """Makes the gradient function for a function fn acting on SU(N) elements.\n
    The gradient function returns coefficients for the su(N) generators.
    """
    return jax.jit(lambda U, *args, **kwargs: jax.grad(lambda w: func(LALG_SU_N(w) @ U, *args, **kwargs))(jnp.zeros((*U.shape[:-2], U.shape[-1]*U.shape[-1]-1))))

def _make_generators(N):
    def _alpha_index(n, m):
        assert m < n
        return n*n + 2*(m-n) - 2

    def _beta_index(n, m):
        assert m < n
        return n*n + 2*(m-n) - 1

    def _gamma_index(n):
        return n*n - 2

    result = np.empty((N*N-1, N, N), dtype=np.complex128)
    for m in range(1, N+1):
        for n in range(m+1, N+1):
            gen = np.zeros((N, N), dtype=np.complex128)
            gen[m-1, n-1] = gen[n-1, m-1] = 0.5
            result[_alpha_index(n, m)] = gen

            gen = np.zeros((N, N), dtype=np.complex128)
            gen[m-1, n-1] = -0.5j
            gen[n-1, m-1] = 0.5j
            result[_beta_index(n, m)] = gen

            gen = np.diag(np.array([1,]*(n-1)+[0,]*(N+1-n), dtype=np.complex128))
            gen[n-1, n-1] = 1-n
            gen /= np.sqrt(2*n*(n-1))
            result[_gamma_index(n)] = gen
    return result
