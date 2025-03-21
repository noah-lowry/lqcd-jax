from functools import partial
import numpy as np

import jax
import jax.numpy as jnp

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

@jax.jit
def wilson_action(field, beta):
    N = field.shape[-1]

    def plaquette(mu, nu):
        U_mu = field[..., mu, :, :]
        U_nu = field[..., nu, :, :]

        U_mu_shifted = jnp.roll(U_mu, shift=-1, axis=nu)
        U_nu_shifted = jnp.roll(U_nu, shift=-1, axis=mu)

        Re_Tr_Plaquettes = jnp.einsum("...AB,...BC,...CD,...DA->...",
                                    U_mu, U_nu_shifted,
                                    U_mu_shifted.conj().mT,
                                    U_nu.conj().mT).real
        
        return 1 - Re_Tr_Plaquettes / N
    
    S = sum(jnp.sum(plaquette(mu, nu)) for mu in range(4) for nu in range(mu+1, 4))

    return beta * S

# avoids big numbers, therefore reduces floating point errors
@jax.jit
def accurate_wilson_hamiltonian_error(q0, p0, q1, p1, beta):

    field0 = LALG_SU_N(q0)
    field1 = LALG_SU_N(q1)

    def plaquette(field, mu, nu):
        U_mu = field[..., mu, :, :]
        U_nu = field[..., nu, :, :]

        U_mu_shifted = jnp.roll(U_mu, shift=-1, axis=nu)
        U_nu_shifted = jnp.roll(U_nu, shift=-1, axis=mu)

        Re_Tr_Plaquettes = jnp.einsum("...AB,...BC,...CD,...DA->...",
                                    U_mu, U_nu_shifted,
                                    U_mu_shifted.conj().mT,
                                    U_nu.conj().mT).real

        return Re_Tr_Plaquettes
    
    def plaquette_diff(mu, nu):
        return (plaquette(field0, mu, nu) - plaquette(field1, mu, nu)) / 3
    
    local_hamiltonian_energy_diff = beta * sum(plaquette_diff(mu, nu) for mu in range(4) for nu in range(mu+1, 4)) + jnp.sum((p1 - p0) * ((p1 + p0) / 2), axis=[-1, -2])

    return jnp.sum(local_hamiltonian_energy_diff)

@jax.jit
def tree_level_improved_action(field, beta):

    def P(mu, nu):
        U_mu = field[..., mu, :, :]
        U_nu = field[..., nu, :, :]

        P = jnp.einsum("...AB,...BC,...CD,...DA->...",
                                    U_mu,
                                    jnp.roll(U_nu, shift=-1, axis=mu),
                                    jnp.roll(U_mu, shift=-1, axis=nu).conj().mT,
                                    U_nu.conj().mT).real
        
        return P
    
    def R(mu, nu):
        U_mu = field[..., mu, :, :]
        U_nu = field[..., nu, :, :]

        R1 = jnp.einsum("...AB,...BC,...CD,...DE,...EF,...FA->...",
                                    U_mu,
                                    jnp.roll(U_nu, shift=-1, axis=mu),
                                    jnp.roll(U_nu, shift=[-1, -1], axis=[mu, nu]),
                                    jnp.roll(U_mu, shift=-2, axis=nu).conj().mT,
                                    jnp.roll(U_nu, shift=-1, axis=nu).conj().mT,
                                    U_nu.conj().mT).real
        
        R2 = jnp.einsum("...AB,...BC,...CD,...DE,...EF,...FA->...",
                                    U_mu,
                                    jnp.roll(U_mu, shift=-1, axis=mu),
                                    jnp.roll(U_nu, shift=-2, axis=mu),
                                    jnp.roll(U_mu, shift=[-1, -1], axis=[mu, nu]).conj().mT,
                                    jnp.roll(U_mu, shift=-1, axis=nu).conj().mT,
                                    U_nu.conj().mT).real
        
        return R1 + R2
    
    u0_sqr = jnp.power(jnp.array([P(mu, nu) for mu in range(4) for nu in range(4) if mu != nu]).mean() / 3, 1/2)
    jax.debug.print("u0^2 = {u}", u=u0_sqr)

    S = sum(
        (5*beta) * (1 - P(mu, nu).sum() / 3) - (beta/(4*u0_sqr)) * (1 - R(mu, nu).sum() / 3)
        for mu in range(4) for nu in range(mu+1, 4)
    )

    return S

@partial(jax.jit, static_argnums=(1, 2))
def wilson_loops_range(field, R, T):
    
    def _wilson_lines(field, L, mu):
        field = field[..., mu, :, :]

        def scan1(carry, i):
            carry = carry @ jnp.roll(field, shift=-i, axis=mu)
            return carry, carry
        
        _, result = jax.lax.scan(
            scan1,
            init=jnp.broadcast_to(jnp.eye(3, dtype=field.dtype), field.shape),
            xs=jnp.arange(L)
        )
        return result

    def _wilson_looper_fn(lr, lt, Ur, Ut, rdim, tdim):
        result = jnp.einsum(
            "...AB,...BC,...CD,...DA->...",
            Ur[lr],
            jnp.roll(Ut[lt], shift=-(lr+1), axis=rdim),
            jnp.roll(Ur[lr], shift=-(lt+1), axis=tdim).conj().mT,
            Ut[lt].conj().mT
        )
        return result
        
    def _wilson_loops_mn(field, R, T, spatial_dim, temporal_dim):
        Ur = _wilson_lines(field, R, spatial_dim)
        Ut = _wilson_lines(field, T, temporal_dim)

        f_over_lt = jax.vmap(lambda lt, lr: _wilson_looper_fn(lr, lt, Ur, Ut, spatial_dim, temporal_dim).mean(), in_axes=(0, None))
        v_f = jax.vmap(lambda lr: f_over_lt(jnp.arange(T), lr), in_axes=0)

        return v_f(jnp.arange(R))
    
    f_mn = lambda mu, nu: _wilson_loops_mn(field, R, T, mu, nu)
    result = sum(f_mn(n, m) for m, n in zip(*np.triu_indices(4, k=1))) / 6  # where m is the time dimension
    return result

    
@jax.jit
def smear_HYP(field, alpha1=0.75, alpha2=0.6, alpha3=0.3):

    def staple_func(f1, f2, d1, d2):
        
        S_plus = jnp.einsum(
            "...ij,...jk,...kl->...il",
            f2,
            jnp.roll(f1, shift=-1, axis=d2),
            jnp.roll(f2, shift=-1, axis=d1).conj().mT
        )
        
        S_minus = jnp.einsum(
            "...ij,...jk,...kl->...il",
            jnp.roll(f2, shift=1, axis=d2).conj().mT,
            jnp.roll(f1, shift=1, axis=d2),
            jnp.roll(f2, shift=[1, -1], axis=[d2, d1])
        )

        return S_plus + S_minus
    
    def V_bar_mu_nu_rho(original_links, mu, nu, rho, alpha3):
        staples = sum(staple_func(original_links[..., mu, :, :], original_links[..., eta, :, :], mu, eta)
                for eta in range(4) if eta != rho and eta != nu and eta != mu)
        vbar = (1 - alpha3) * original_links[..., mu, :, :] + (alpha3 / 2) * staples
        vbar = proj_SU3(vbar)
        return vbar
    
    def V_tilda_mu_nu(original_links, mu, nu, alpha2, alpha3):
        staples = sum(staple_func(V_bar_mu_nu_rho(original_links, mu, rho, nu, alpha3), V_bar_mu_nu_rho(original_links, rho, nu, mu, alpha3), mu, rho)
                for rho in range(4) if rho != nu and rho != mu)
        vtilda = (1 - alpha2) * original_links[..., mu, :, :] + (alpha2 / 4) * staples
        vtilda = proj_SU3(vtilda)
        return vtilda
    
    def V_final_mu(original_links, mu, alpha1, alpha2, alpha3):
        staples = sum(staple_func(V_tilda_mu_nu(original_links, mu, nu, alpha2, alpha3), V_tilda_mu_nu(original_links, nu, mu, alpha2, alpha3), mu, nu)
                for nu in range(4) if nu != mu)
        v = (1 - alpha1) * original_links[..., mu, :, :] + (alpha1 / 6) * staples
        v = proj_SU3(v)
        return v

    U_HYP = jnp.stack([V_final_mu(field, mu, alpha1, alpha2, alpha3) for mu in range(4)], axis=-3)

    return U_HYP
