from functools import partial
import numpy as np

import jax
import jax.numpy as jnp

from special_unitary import proj_SU3, expi_H3_util

@partial(jax.jit, static_argnums=(1, 2))
def _plaquette_mn(U, mu, nu):
    """Trace real plaquette divided by N everywhere in the mu nu plane"""
    N = U.shape[-1]

    U_mu = U[..., mu, :, :]
    U_nu = U[..., nu, :, :]

    U_mu_shifted = jnp.roll(U_mu, shift=-1, axis=nu)
    U_nu_shifted = jnp.roll(U_nu, shift=-1, axis=mu)

    Re_Tr_Plaquettes = jnp.einsum("...AB,...BC,...CD,...DA->...",
                                U_mu, U_nu_shifted,
                                U_mu_shifted.conj().mT,
                                U_nu.conj().mT).real
    
    return Re_Tr_Plaquettes / N

@jax.jit
def wilson_action(links, beta):
    S = sum(jnp.sum(1 - _plaquette_mn(links, mu, nu)) for mu in range(4) for nu in range(mu+1, 4))
    return beta * S

# avoids big numbers, therefore reduces floating point errors
@jax.jit
def accurate_wilson_hamiltonian_error(q0, p0, q1, p1, beta):

    local_hamiltonian_energy_diff = beta * sum(
        _plaquette_mn(q0, mu, nu) - _plaquette_mn(q1, mu, nu)
        for mu in range(4)
        for nu in range(mu + 1, 4)
    ) + jnp.sum((p1 - p0) * ((p1 + p0) / 2), axis=[-1, -2])

    return jnp.sum(local_hamiltonian_energy_diff)

@partial(jax.jit, static_argnames=("u0",))
def luscher_weisz_action(links, beta, u0=None):
    """`beta` is the canonical beta. `beta_LW` is computed as `u0^-4 * beta * c_pl`. If `u0` is not provided, it is calculated from the configuration."""

    def _rect_mn(U, mu, nu):
        N = U.shape[-1]
        
        U_mu = U[..., mu, :, :]
        U_nu = U[..., nu, :, :]

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
        
        return (R1 + R2) / N
    
    def _pgram_mn(U, mu, nu, rho):
        N = U.shape[-1]

        U_mu = U[..., mu, :, :]
        U_nu = U[..., nu, :, :]
        U_rho = U[..., rho, :, :]

        PG = jnp.einsum(
            "...AB,...BC,...CD,...DE,...EF,...FA->...",
            U_mu,
            jnp.roll(U_rho, shift=-1, axis=mu),
            jnp.roll(U_nu, shift=[-1, -1], axis=[mu, rho]),
            jnp.roll(U_mu, shift=[-1, -1], axis=[nu, rho]).conj().mT,
            jnp.roll(U_rho, shift=-1, axis=nu).conj().mT,
            U_nu.conj().mT
        ).real
        
        return PG / N

    plaq = jnp.stack([_plaquette_mn(links, mu, nu) for mu in range(4) for nu in range(mu+1, 4)], axis=0)
    rect = jnp.stack([_rect_mn(links, mu, nu) for mu in range(4) for nu in range(mu+1, 4)], axis=0)
    pgram = jnp.stack([_pgram_mn(links, mu, nu, rho) for mu in range(4) for nu in range(mu+1, 4) for rho in range(nu+1, 4)], axis=0)

    u0 = plaq.mean() ** 0.25 if u0 is None else u0

    alpha_s = -1.303615*jnp.log(u0)
    
    S_local = beta * (
        (1 - plaq).sum(axis=0) \
        - ((1 + 0.4805*alpha_s) / (20 * u0**2)) * (1 - rect).sum(axis=0) \
        - (0.03325*alpha_s / (u0**2)) * (1 - pgram).sum(axis=0)
    )
    
    return jnp.sum(S_local)

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
    result = sum(f_mn(n, m) for m, n in [(0, 1), (0, 2), (0, 3)]) / 3  # where m is the time dimension
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

@partial(jax.jit, static_argnums=(1, 2, 3))
def smear_stout(links, n=10, rho=0.1, temporal=False):

    N = links.shape[-1]
    
    rho_matrix = np.full(shape=(4, 4), fill_value=rho)
    if not temporal:
        rho_matrix[:, 0] = 0
        rho_matrix[0, :] = 0

    def _staples_mn(U, mu, nu):
        U_mu = U[..., mu, :, :]
        U_nu = U[..., nu, :, :]

        S_plus = jnp.einsum(
            "...ij,...jk,...kl->...il",
            U_nu,
            jnp.roll(U_mu, shift=-1, axis=nu),
            jnp.roll(U_nu, shift=-1, axis=mu).conj().mT
        )
        
        S_minus = jnp.einsum(
            "...ij,...jk,...kl->...il",
            jnp.roll(U_nu, shift=1, axis=nu).conj().mT,
            jnp.roll(U_mu, shift=1, axis=nu),
            jnp.roll(U_mu, shift=[1, -1], axis=[nu, mu])
        )

        return S_plus + S_minus
    
    def kernel(_, U):
        staples = jnp.stack([sum(rho_matrix[mu, nu] * _staples_mn(U, mu, nu) for nu in range(4) if nu != mu) for mu in range(4)], axis=-3)

        Omega = staples @ U.conj().mT
        O = Omega.conj().mT - Omega
        Q = 0.5j * O - 0.5j * (jnp.trace(O, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)) / N

        result = expi_H3_util(Q) @ U
        return result
    
    result = jax.lax.fori_loop(0, n, kernel, links)

    return result
