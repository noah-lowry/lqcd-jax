import jax
from gauge_field_utils import LALG_SU_N

def int_MN4_takaishi_forcrand(q0, p0, F_func, tau, nfev_approx, theta=0.08398315262876693, rho=0.2539785108410595, lambd=0.6822365335719091, mu=-0.03230286765269967):
    eps = tau / (nfev_approx // 5)

    def scan_fn(carry, _):
        T_operator = lambda q, p, coef: LALG_SU_N(coef * p) @ q
        q, p = carry

        p = p - theta * eps * F_func(q)
        q = T_operator(q, p, rho * eps)
        p = p - lambd * eps * F_func(q)
        q = T_operator(q, p, mu * eps)
        p = p - (1 - 2 * (lambd + theta)) * 0.5 * eps * F_func(q)
        q = T_operator(q, p, (1 - 2 * (mu + rho)) * eps)
        p = p - (1 - 2 * (lambd + theta)) * 0.5 * eps * F_func(q)
        q = T_operator(q, p, mu * eps)
        p = p - lambd * eps * F_func(q)
        q = T_operator(q, p, rho * eps)
        p = p - theta * eps * F_func(q)

        return (q, p), None
    
    (qt, pt), _ = jax.lax.scan(
        f=scan_fn,
        init=(q0, p0),
        length=nfev_approx//5
    )
        
    return qt, pt

def int_MN2_omelyan(q0, p0, F_func, tau, nfev_approx, lambd=0.1931833275037836):
    eps = tau / (nfev_approx // 2)

    def scan_fn(carry, _):
        T_operator = lambda q, p, coef: LALG_SU_N(coef * p) @ q
        q, p = carry
        
        q = T_operator(q, p, lambd * eps)
        p = p - 0.5 * eps * F_func(q)
        q = T_operator(q, p, (1 - 2 * lambd) * eps)
        p = p - 0.5 * eps * F_func(q)
        q = T_operator(q, p, lambd * eps)

        return (q, p), None
    
    (qt, pt), _ = jax.lax.scan(
        f=scan_fn,
        init=(q0, p0),
        length=nfev_approx//2
    )

    return qt, pt

def int_LF2(q0, p0, F_func, tau, nfev_approx):
    eps = tau / nfev_approx


    def scan_fn(carry, _):
        T_operator = lambda q, p, coef: LALG_SU_N(coef * p) @ q

        q, p = carry
        p = p - 0.5 * eps * F_func(q)
        q = T_operator(q, p, eps)
        p = p - 0.5 * eps * F_func(q)
        return (q, p), None
    
    (qt, pt), _ = jax.lax.scan(
        f=scan_fn,
        init=(q0, p0),
        length=nfev_approx
    )

    return qt, pt
