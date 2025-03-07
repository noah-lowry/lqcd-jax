from functools import partial

import jax
import jax.numpy as jnp
from sun_utils import SUGenerator

gen = SUGenerator(3, module="jax", dtype=jnp.complex64)
@jax.jit
def coef_to_lie_group(coef):
    su = jnp.einsum("...N,Nij->...ij", coef.astype(jnp.complex64), gen.generators)
    SU = jax.scipy.linalg.expm(1j*su)
    return SU

@partial(jax.jit, static_argnames=["beta"])
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

@partial(jax.jit, static_argnames=["beta"])
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