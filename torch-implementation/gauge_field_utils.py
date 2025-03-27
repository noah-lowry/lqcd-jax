import numpy as np

import torch
import torch.linalg

from special_unitary import proj_SU3

def wilson_action(links, beta):
    S = sum(torch.sum(1 - _plaquette_mn(links, mu, nu)) for mu in range(4) for nu in range(mu+1, 4))
    return beta * S

def wilson_gauge_error(q0, p0, q1, p1, beta):

    local_hamiltonian_energy_diff = beta * sum(
        _plaquette_mn(q0, mu, nu) - _plaquette_mn(q1, mu, nu)
        for mu in range(4)
        for nu in range(mu + 1, 4)
    ) + torch.sum((p1 - p0) * ((p1 + p0) / 2), dim=[-1, -2])

    return torch.sum(local_hamiltonian_energy_diff)

def luscher_weisz_action(links, beta, u0):
    """`beta` is beta_LW (aka beta_pl)"""

    plaq = torch.stack([_plaquette_mn(links, mu, nu) for mu in range(4) for nu in range(mu+1, 4)], dim=0)
    rect = torch.stack([_rect_mn(links, mu, nu) for mu in range(4) for nu in range(mu+1, 4)], dim=0)
    pgram = torch.stack([_pgram_mn(links, mu, nu, rho) for mu in range(4) for nu in range(mu+1, 4) for rho in range(nu+1, 4)], dim=0)

    alpha_s = -1.303615*np.log(u0)
    
    S_local = beta * (
        (1 - plaq).sum(dim=0) \
        - ((1 + 0.4805*alpha_s) / (20 * u0**2)) * (1 - rect).sum(dim=0) \
        - (0.03325*alpha_s / (u0**2)) * (1 - pgram).sum(dim=0)
    )
    
    return torch.sum(S_local)

def luscher_weisz_gauge_error(q0, p0, q1, p1, beta, u0):
    
    plaq = torch.stack([_plaquette_mn(q1, mu, nu)-_plaquette_mn(q0, mu, nu) for mu in range(4) for nu in range(mu+1, 4)], dim=0).sum(dim=0)
    rect = torch.stack([_rect_mn(q1, mu, nu)-_rect_mn(q0, mu, nu) for mu in range(4) for nu in range(mu+1, 4)], dim=0).sum(dim=0)
    pgram = torch.stack([_pgram_mn(q1, mu, nu, rho)-_pgram_mn(q0, mu, nu, rho) for mu in range(4) for nu in range(mu+1, 4) for rho in range(nu+1, 4)], dim=0).sum(dim=0)

    alpha_s = -1.303615*torch.log(u0)

    S_local = beta * (
        -plaq \
        + ((1 + 0.4805*alpha_s) / (20 * u0**2)) * rect \
        + (0.03325*alpha_s / (u0**2)) * pgram
    )

    change_local = S_local + torch.multiply(p1 - p0, (p1 + p0) / 2).sum(dim=(-2, -1))

    return torch.sum(change_local)

def wilson_loops_range(field, R, T):
    
    def _wilson_lines(field, L, mu):
        field = field[..., mu, :, :]

        def scan1(carry, i):
            carry = carry @ torch.roll(field, shifts=-i, axis=mu)
            return carry, carry
        
        result = []
        carry = torch.broadcast_to(torch.eye(3, dtype=field.dtype), field.shape)
        for x in torch.arange(L):
            carry, y = scan1(carry, x)
            result.append(y)
        result = torch.stack(result)

        return result

    def _wilson_looper_fn(lr, lt, Ur, Ut, rdim, tdim):
        result = torch.einsum(
            "...AB,...BC,...CD,...DA->...",
            Ur[lr],
            torch.roll(Ut[lt], shifts=-(lr+1), axis=rdim),
            torch.roll(Ur[lr], shifts=-(lt+1), axis=tdim).mH,
            Ut[lt].mH
        )
        return result
        
    def _wilson_loops_mn(field, R, T, spatial_dim, temporal_dim):
        Ur = _wilson_lines(field, R, spatial_dim)
        Ut = _wilson_lines(field, T, temporal_dim)

        f_over_lt = torch.func.vmap(lambda lt, lr: _wilson_looper_fn(lr, lt, Ur, Ut, spatial_dim, temporal_dim).mean(), in_dims=(0, None))
        v_f = torch.func.vmap(lambda lr: f_over_lt(torch.arange(T), lr), in_dims=0)

        return v_f(torch.arange(R))
    
    f_mn = lambda mu, nu: _wilson_loops_mn(field, R, T, mu, nu)
    result = sum(f_mn(n, m) for m, n in [(0, 1), (0, 2), (0, 3)]) / 3  # where m is the time dimension
    return result

def smear_HYP(field, alpha1=0.75, alpha2=0.6, alpha3=0.3):

    def staple_func(f1, f2, d1, d2):
        
        S_plus = torch.einsum(
            "...ij,...jk,...kl->...il",
            f2,
            torch.roll(f1, shifts=-1, dims=d2),
            torch.roll(f2, shifts=-1, dims=d1).mH
        )
        
        S_minus = torch.einsum(
            "...ij,...jk,...kl->...il",
            torch.roll(f2, shifts=1, dims=d2).mH,
            torch.roll(f1, shifts=1, dims=d2),
            torch.roll(f2, shifts=[1, -1], dims=[d2, d1])
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

    U_HYP = torch.stack([V_final_mu(field, mu, alpha1, alpha2, alpha3) for mu in range(4)], dim=-3)

    return U_HYP

def smear_stout(links, n=10, rho=0.1, temporal=False):

    N = links.shape[-1]
    
    rho_matrix = np.full(shape=(4, 4), fill_value=rho)
    if not temporal:
        rho_matrix[:, 0] = 0
        rho_matrix[0, :] = 0

    def _staples_mn(U, mu, nu):
        U_mu = U[..., mu, :, :]
        U_nu = U[..., nu, :, :]

        S_plus = torch.einsum(
            "...ij,...jk,...kl->...il",
            U_nu,
            torch.roll(U_mu, shifts=-1, dims=nu),
            torch.roll(U_nu, shifts=-1, dims=mu).mH
        )
        
        S_minus = torch.einsum(
            "...ij,...jk,...kl->...il",
            torch.roll(U_nu, shifts=1, dims=nu).mH,
            torch.roll(U_mu, shifts=1, dims=nu),
            torch.roll(U_mu, shifts=[1, -1], dims=[nu, mu])
        )

        return S_plus + S_minus
    
    def kernel(_, U):
        staples = torch.stack([sum(rho_matrix[mu, nu] * _staples_mn(U, mu, nu) for nu in range(4) if nu != mu) for mu in range(4)], axis=-3)

        Omega = staples @ U.mH
        O = Omega.mH - Omega
        Q = 0.5j * O - 0.5j * (torch.einsum("...ii->...", O)[..., None, None] * torch.eye(3, dtype=O.dtype, device=O.device)) / N

        result = torch.linalg.matrix_exp(1j*Q) @ U
        return result
    
    result = links
    for i in range(n):
        result = kernel(i, result)

    return result

def mean_plaquette(links):
    N = links.shape[-1]
    return torch.stack([
        N*_plaquette_mn(links, mu, nu)
        for mu in range(4) for nu in range(mu+1, 4)
    ]).mean()

def _plaquette_mn(U, mu, nu):
    """Trace real plaquette divided by N everywhere in the mu nu plane"""
    N = U.shape[-1]

    U_mu = U[..., mu, :, :]
    U_nu = U[..., nu, :, :]

    U_mu_shifted = torch.roll(U_mu, shifts=-1, dims=nu)
    U_nu_shifted = torch.roll(U_nu, shifts=-1, dims=mu)

    Re_Tr_Plaquettes = torch.einsum("...AB,...BC,...CD,...DA->...",
                                U_mu, U_nu_shifted,
                                U_mu_shifted.mH,
                                U_nu.mH).real
    
    return Re_Tr_Plaquettes / N

def _rect_mn(U, mu, nu):
    N = U.shape[-1]
    
    U_mu = U[..., mu, :, :]
    U_nu = U[..., nu, :, :]

    R1 = torch.einsum("...AB,...BC,...CD,...DE,...EF,...FA->...",
                                U_mu,
                                torch.roll(U_nu, shifts=-1, dims=mu),
                                torch.roll(U_nu, shifts=[-1, -1], dims=[mu, nu]),
                                torch.roll(U_mu, shifts=-2, dims=nu).mH,
                                torch.roll(U_nu, shifts=-1, dims=nu).mH,
                                U_nu.mH).real
    
    R2 = torch.einsum("...AB,...BC,...CD,...DE,...EF,...FA->...",
                                U_mu,
                                torch.roll(U_mu, shifts=-1, dims=mu),
                                torch.roll(U_nu, shifts=-2, dims=mu),
                                torch.roll(U_mu, shifts=[-1, -1], dims=[mu, nu]).mH,
                                torch.roll(U_mu, shifts=-1, dims=nu).mH,
                                U_nu.mH).real
    
    return (R1 + R2) / N

def _pgram_mn(U, mu, nu, rho):
    N = U.shape[-1]

    U_mu = U[..., mu, :, :]
    U_nu = U[..., nu, :, :]
    U_rho = U[..., rho, :, :]

    PG1 = torch.einsum(
        "...AB,...BC,...CD,...DE,...EF,...FA->...",
        U_mu,
        torch.roll(U_rho, shifts=-1, dims=mu),
        torch.roll(U_nu, shifts=[-1, -1], dims=[mu, rho]),
        torch.roll(U_mu, shifts=[-1, -1], dims=[nu, rho]).mH,
        torch.roll(U_rho, shifts=-1, dims=nu).mH,
        U_nu.mH
    ).real

    PG2 = torch.einsum(
        "...AB,...BC,...CD,...DE,...EF,...FA->...",
        U_mu,
        torch.roll(U_nu, shifts=-1, dims=mu),
        torch.roll(U_rho, shifts=[-1, -1], dims=[mu, nu]),
        torch.roll(U_mu, shifts=[-1, -1], dims=[nu, rho]).mH,
        torch.roll(U_nu, shifts=-1, dims=rho).mH,
        U_rho.mH
    ).real
    
    PG3 = torch.einsum(
        "...AB,...BC,...CD,...DE,...EF,...FA->...",
        U_nu,
        torch.roll(U_mu, shifts=-1, dims=nu),
        torch.roll(U_rho, shifts=[-1, -1], dims=[mu, nu]),
        torch.roll(U_nu, shifts=[-1, -1], dims=[mu, rho]).mH,
        torch.roll(U_mu, shifts=-1, dims=rho).mH,
        U_rho.mH
    ).real

    PG4 = torch.einsum(
        "...AB,...BC,...CD,...DE,...EF,...FA->...",
        torch.roll(U_mu, shifts=-1, dims=nu),
        torch.roll(U_nu, shifts=-1, dims=mu).mH,
        torch.roll(U_rho, shifts=-1, dims=mu),
        torch.roll(U_mu, shifts=-1, dims=rho).mH,
        torch.roll(U_nu, shifts=-1, dims=rho),
        torch.roll(U_rho, shifts=-1, dims=nu).mH
    ).real
    
    return (PG1 + PG2 + PG3 + PG4) / N
