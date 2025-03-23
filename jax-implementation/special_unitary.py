from functools import partial
import numpy as np

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(1,))
def LALG_SU_N(coef, N=3):
    """Takes as input a float array `coef[..., N^2-1]` and returns complex array `SU[..., N, N]`.
    The N^2-1 values in `coef` correspond to the coefficients of su(N) generators (e.g. N=3 corresponds to the Gell-Mann matrices divided by 2)\n
    Lie Algebra to Lie Group Special Unitary N."""
    
    su = _LA_SU_N(coef, N)
    SU = jax.scipy.linalg.expm(1j*su)

    return SU

@partial(jax.jit, static_argnums=(1,))
def _LA_SU_N(coef, N=3):
    """Takes as input a float (or real-complex) array `coef[..., N^2-1]` and returns complex array `su[..., N, N]` such that `expm(1j*su)` belongs to SU(N)."""
    coef = jax.lax.complex(coef.real, jnp.zeros_like(coef.real))
    gmm = _make_generators(N)
    su = jnp.einsum("...N,Nij->...ij", coef, gmm)
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

    violation = jnp.linalg.matrix_norm(jnp.matmul(arr.conj().mT, arr, precision=jax.lax.Precision.HIGHEST) - I, ord="fro")

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
    return jax.jit(lambda U, *args, **kwargs: jax.grad(lambda w: func(fast_expi_su3(w) @ U, *args, **kwargs))(jnp.zeros((*U.shape[:-2], U.shape[-1]*U.shape[-1]-1))))

@jax.custom_jvp
@jax.jit
def fast_expi_su3(q):
    """Computes `exp(i qa*λa)` where `qa` are coefficients of the Gell-Mann matrices λa.\n
    This implementation is much more efficient than using jax.scipy.linalg.expm."""

    gmm = _make_generators(3)
    q = jax.lax.complex(q.real, jnp.zeros_like(q.real))
    Q = jnp.einsum("...N,Nij->...ij", q, gmm, precision=jax.lax.Precision.HIGHEST)

    Q__2 = jnp.matmul(Q, Q, precision=jax.lax.Precision.HIGHEST)

    c0 = jnp.einsum("...AB,...BA->...", Q__2, Q, precision=jax.lax.Precision.HIGHEST).real / 3
    c1_3 = jnp.trace(Q__2, axis1=-2, axis2=-1).real / 6
    c0_max = jax.lax.pow(c1_3, 1.5) * 2

    theta = jax.lax.acos(
        jax.lax.select(
            jnp.isclose(c0_max, 0), jnp.ones_like(c0_max), jax.lax.abs(c0) / c0_max
        )  # Tr(Q^2) == 0 -> Tr(Q^3) == 0 (additionally, Tr(Q^2) == 0 -> Q == 0)
    )
    u__2 = c1_3 * jax.lax.cos(theta / 3)**2
    w__2 = (c1_3 - u__2) * 3

    u = jax.lax.sqrt(u__2)
    w = jax.lax.sqrt(w__2)

    cos_w = jax.lax.cos(w)
    sinc_w = jnp.sinc(w / jnp.pi)
    exp_miu = jax.lax.exp(-1j*u)
    exp_2iu = 1 / (exp_miu * exp_miu)

    h0 = (u__2 - w__2) * exp_2iu + exp_miu * (8*u__2*cos_w + 2j*u*(3*u__2 + w__2)*sinc_w)
    h1 = 2*u*exp_2iu - exp_miu * (2*u*cos_w - 1j*(3*u__2 - w__2)*sinc_w)
    h2 = exp_2iu - exp_miu * (cos_w + 3j*u*sinc_w)

    dd = 9*u__2 - w__2

    f0 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(h0), h0 / dd)
    f1 = jax.lax.select(jnp.isclose(dd, 0), jnp.full_like(h1, 1j), h1 / dd)
    f2 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(h2), h2 / dd)

    f0 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(f0), f0)
    f1 = jax.lax.select(jnp.signbit(c0), -jnp.conjugate(f1), f1)
    f2 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(f2), f2)

    I = jnp.broadcast_to(jnp.eye(3), Q.shape)
    result = f0[..., None, None]*I + f1[..., None, None]*Q + f2[..., None, None]*Q__2

    return result

@fast_expi_su3.defjvp
@jax.jit
def _fast_expi_su3_frechet(q, t):
    q, = q
    t, = t
    q = jax.lax.complex(q.real, jnp.zeros_like(q.real))
    t = jax.lax.complex(t.real, jnp.zeros_like(t.real))
    gmm = _make_generators(3)
    Q = jnp.einsum("...N,Nij->...ij", q, gmm)
    T = jnp.einsum("...N,Nij->...ij", t, gmm)

    Q__2 = jnp.matmul(Q, Q, precision=jax.lax.Precision.HIGHEST)
    c0 = jnp.einsum("...AB,...BA->...", Q__2, Q, precision=jax.lax.Precision.HIGHEST).real / 3
    c1_3 = jnp.trace(Q__2, axis1=-2, axis2=-1).real / 6
    c0_max = jax.lax.pow(c1_3, 1.5) * 2

    theta = jax.lax.acos(
        jax.lax.select(
            jnp.isclose(c0_max, 0), jnp.ones_like(c0_max), jax.lax.abs(c0) / c0_max
        )  # Tr(Q^2) == 0 -> Tr(Q^3) == 0 (additionally, Tr(Q^2) == 0 -> Q == 0)
    )
    u__2 = c1_3 * jax.lax.cos(theta / 3)**2
    w__2 = (c1_3 - u__2) * 3

    u = jax.lax.sqrt(u__2)
    w = jax.lax.sqrt(w__2)

    cos_w = jax.lax.cos(w)
    sinc_w = jnp.sinc(w / jnp.pi)
    exp_miu = jax.lax.exp(-1j*u)
    exp_2iu = 1 / (exp_miu * exp_miu)

    h0 = (u__2 - w__2) * exp_2iu + exp_miu * (8*u__2*cos_w + 2j*u*(3*u__2 + w__2)*sinc_w)
    h1 = 2*u*exp_2iu - exp_miu * (2*u*cos_w - 1j*(3*u__2 - w__2)*sinc_w)
    h2 = exp_2iu - exp_miu * (cos_w + 3j*u*sinc_w)

    dd = 9*u__2 - w__2
    
    f0 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(h0), h0 / dd)
    f1 = jax.lax.select(jnp.isclose(dd, 0), jnp.full_like(h1, 1j), h1 / dd)
    f2 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(h2), h2 / dd)

    I = jnp.broadcast_to(jnp.eye(3), Q.shape)
    expi = jax.lax.select(jnp.signbit(c0), jnp.conjugate(f0), f0)[..., None, None]*I \
                    + jax.lax.select(jnp.signbit(c0), -jnp.conjugate(f1), f1)[..., None, None]*Q \
                    + jax.lax.select(jnp.signbit(c0), jnp.conjugate(f2), f2)[..., None, None]*Q__2

    xi1 = (cos_w - sinc_w) / w__2

    r01 = 2*(u+1j*(u__2-w__2))*exp_2iu + 2*exp_miu*(4*u*(2-1j*u)*cos_w + 1j*(9*u__2 + w__2 - 1j*u*(3*u__2+w__2))*sinc_w)
    r11 = 2*(1+2j*u)*exp_2iu + exp_miu*(-2*(1-1j*u)*cos_w + 1j*(6*u+1j*(w__2-3*u__2))*sinc_w)
    r21 = 2j*exp_2iu + 1j*exp_miu*(cos_w - 3*(1-1j*u)*sinc_w)
    r02 = -2*exp_2iu + 2j*u*exp_miu*(cos_w + (1+4j*u)*sinc_w + 3*u__2*xi1)
    r12 = -1j*exp_miu*(cos_w + (1+2j*u)*sinc_w - 3*u__2*xi1)
    r22 = exp_miu*(sinc_w - 3j*u*xi1)
    
    ddd = 2*dd**2

    b10 = (2*u*r01 + (3*u__2 - w__2)*r02 - 2*(15*u__2 + w__2)*f0)
    b10 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(b10), b10 / ddd)
    b10 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(b10), b10)

    b11 = (2*u*r11 + (3*u__2 - w__2)*r12 - 2*(15*u__2 + w__2)*f1)
    b11 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(b11), b11 / ddd)
    b11 = jax.lax.select(jnp.signbit(c0), -jnp.conjugate(b11), b11)

    b12 = (2*u*r21 + (3*u__2 - w__2)*r22 - 2*(15*u__2 + w__2)*f2)
    b12 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(b12), b12 / ddd)
    b12 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(b12), b12)

    b20 = (r01 - 3*u*r02 - 24*u*f0)
    b20 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(b20), b20 / ddd)
    b20 = jax.lax.select(jnp.signbit(c0), -jnp.conjugate(b20), b20)

    b21 = (r11 - 3*u*r12 - 24*u*f1)
    b21 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(b21), b21 / ddd)
    b21 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(b21), b21)

    b22 = (r21 - 3*u*r22 - 24*u*f2)
    b22 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(b22), b22 / ddd)
    b22 = jax.lax.select(jnp.signbit(c0), -jnp.conjugate(b22), b22)


    B1 = b10[..., None, None]*I + b11[..., None, None]*Q + b12[..., None, None]*Q__2
    B2 = b20[..., None, None]*I + b21[..., None, None]*Q + b22[..., None, None]*Q__2
    
    QT = jnp.matmul(Q, T, precision=jax.lax.Precision.HIGHEST)

    Tr1 = jnp.trace(QT, axis1=-2, axis2=-1)
    Tr2 = jnp.einsum("...AB,...BA->...", Q__2, T, precision=jax.lax.Precision.HIGHEST)

    expi_frechet = Tr1[..., None, None]*B1 + Tr2[..., None, None]*B2 \
                    + jax.lax.select(jnp.signbit(c0), -jnp.conjugate(f1), f1)[..., None, None]*T \
                    + jax.lax.select(jnp.signbit(c0), jnp.conjugate(f2), f2)[..., None, None] \
                        *(jnp.matmul(T, Q, precision=jax.lax.Precision.HIGHEST) + QT)

    return expi, expi_frechet

@jax.custom_jvp
@jax.jit
def expi_H3_util(Q):
    """Compute exp(iQ) for traceless 3x3 Hermetian Q. Do not use this function with gradient calculations"""
    Q__2 = jnp.matmul(Q, Q, precision=jax.lax.Precision.HIGHEST)

    c0 = jnp.einsum("...AB,...BA->...", Q__2, Q, precision=jax.lax.Precision.HIGHEST).real / 3
    c1_3 = jnp.trace(Q__2, axis1=-2, axis2=-1).real / 6
    c0_max = jax.lax.pow(c1_3, 1.5) * 2

    theta = jax.lax.acos(
        jax.lax.select(
            jnp.isclose(c0_max, 0), jnp.ones_like(c0_max), jax.lax.abs(c0) / c0_max
        )  # Tr(Q^2) == 0 -> Tr(Q^3) == 0 (additionally, Tr(Q^2) == 0 -> Q == 0)
    )
    u__2 = c1_3 * jax.lax.cos(theta / 3)**2
    w__2 = (c1_3 - u__2) * 3

    u = jax.lax.sqrt(u__2)
    w = jax.lax.sqrt(w__2)

    cos_w = jax.lax.cos(w)
    sinc_w = jnp.sinc(w / jnp.pi)
    exp_miu = jax.lax.exp(-1j*u)
    exp_2iu = 1 / (exp_miu * exp_miu)

    h0 = (u__2 - w__2) * exp_2iu + exp_miu * (8*u__2*cos_w + 2j*u*(3*u__2 + w__2)*sinc_w)
    h1 = 2*u*exp_2iu - exp_miu * (2*u*cos_w - 1j*(3*u__2 - w__2)*sinc_w)
    h2 = exp_2iu - exp_miu * (cos_w + 3j*u*sinc_w)

    dd = 9*u__2 - w__2

    f0 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(h0), h0 / dd)
    f1 = jax.lax.select(jnp.isclose(dd, 0), jnp.full_like(h1, 1j), h1 / dd)
    f2 = jax.lax.select(jnp.isclose(dd, 0), jnp.ones_like(h2), h2 / dd)

    f0 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(f0), f0)
    f1 = jax.lax.select(jnp.signbit(c0), -jnp.conjugate(f1), f1)
    f2 = jax.lax.select(jnp.signbit(c0), jnp.conjugate(f2), f2)

    I = jnp.broadcast_to(jnp.eye(3), Q.shape)
    result = f0[..., None, None]*I + f1[..., None, None]*Q + f2[..., None, None]*Q__2

    return result

@expi_H3_util.defjvp
@jax.jit
def _give_nan(p, t):
    return jnp.full_like(p[0], jnp.nan), jnp.full_like(t[0], jnp.nan)

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
