from functools import partial

import jax
import jax.numpy as jnp
from sun_utils import SUGenerator

gen = SUGenerator(3, module="jax", dtype=jnp.complex64)
@jax.jit
def coef_to_lie_group(coef):
    coef = jax.lax.complex(coef, jnp.zeros_like(coef))
    su = jnp.einsum("...N,Nij->...ij", coef, gen.generators.astype(coef.dtype))
    SU = jax.scipy.linalg.expm(1j*su)
    return SU

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

    field0 = coef_to_lie_group(q0)
    field1 = coef_to_lie_group(q1)

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

def chain_matmul_einsum(arrs, trace_last=True):
    args = [(arr, [..., i, i+1]) for i, arr in enumerate(arrs)]
    args = [a for b in args for a in b]
    if trace_last:
        args[-1][2] = 0
        args.append([...])
    else:
        args.append([..., 0, len(arrs)])

    result = jnp.einsum(*args)

    return result

@partial(jax.jit, static_argnames=["R", "T", "time_unique"])
def mean_wilson_rectangle(field, R, T, time_unique=True):
    
    result = 0

    if time_unique:
        
        for spatial_dim in [1, 2, 3]:

            link_list = [jnp.roll(field[:,:,:,:,spatial_dim], shift=-i, axis=spatial_dim) for i in range(R)] + \
                            [jnp.roll(field[:,:,:,:,0], shift=[-i, -R], axis=[0, spatial_dim]) for i in range(T)] + \
                            [jnp.roll(field[:,:,:,:,spatial_dim], shift=[-T, -i], axis=[0, spatial_dim]).conj().mT for i in range(R-1, -1, -1)] + \
                            [jnp.roll(field[:,:,:,:,0], shift=-i, axis=0).conj().mT for i in range(T-1, -1, -1)]
            
            result += chain_matmul_einsum(link_list, trace_last=True).mean()
        
        return result / 3
    else:
        
        for mu, nu in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            
            link_list = [jnp.roll(field[:,:,:,:,nu], shift=-i, axis=nu) for i in range(R)] + \
                            [jnp.roll(field[:,:,:,:,mu], shift=[-i, -R], axis=[mu, nu]) for i in range(T)] + \
                            [jnp.roll(field[:,:,:,:,nu], shift=[-T, -i], axis=[mu, nu]).conj().mT for i in range(R-1, -1, -1)] + \
                            [jnp.roll(field[:,:,:,:,mu], shift=-i, axis=mu).conj().mT for i in range(T-1, -1, -1)]
            
            result += chain_matmul_einsum(link_list, trace_last=True).mean()
        
        return result / 6

@jax.jit
def smear_HYP(field, alpha1=0.75, alpha2=0.6, alpha3=0.3):
    
    def proj_SU3(arr):
        D, V = jnp.linalg.eig(arr.conj().mT @ arr)
        result = arr @ ((V * (jnp.expand_dims(D, axis=-2)**-0.5)) @ V.conj().mT)
        result = result * (jnp.expand_dims(jnp.linalg.det(result), axis=[-1, -2]) ** (-1/3))
        return result
    
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
        vbar = (1 - alpha3) * original_links[..., mu, :, :] + (alpha3 / 2) * \
            sum(staple_func(original_links[..., mu, :, :], original_links[..., nu, :, :], mu, eta)
                for eta in range(4) if eta != rho and eta != nu and eta != mu)
        vbar = proj_SU3(vbar)
        return vbar
    
    def V_tilda_mu_nu(original_links, mu, nu, alpha2, alpha3):
        vtilda = (1 - alpha2) * original_links[..., mu, :, :] + (alpha2 / 4) * \
            sum(staple_func(V_bar_mu_nu_rho(original_links, rho, nu, mu, alpha3), V_bar_mu_nu_rho(original_links, mu, rho, nu, alpha3), mu, rho)
                for rho in range(4) if rho != nu and rho != mu)
        vtilda = proj_SU3(vtilda)
        return vtilda
    
    def V_final_mu(original_links, mu, alpha1, alpha2, alpha3):
        v = (1 - alpha1) * original_links[..., mu, :, :] + (alpha1 / 6) * \
            sum(staple_func(V_tilda_mu_nu(original_links, nu, mu, alpha2, alpha3), V_tilda_mu_nu(original_links, mu, nu, alpha2, alpha3), mu, nu)
                for nu in range(4) if nu != mu)
        v = proj_SU3(v)
        return v

    U_HYP = jnp.stack([V_final_mu(field, mu, alpha1, alpha2, alpha3) for mu in range(4)], axis=-3)

    return U_HYP
