from functools import partial

import jax
from special_unitary import fast_expi_su3

@partial(jax.jit, static_argnums=(2, 4))
def int_MN4_takaishi_forcrand(q0, p0, F_func, tau, steps_md, theta=0.08398315262876693, rho=0.2539785108410595, lambd=0.6822365335719091, mu=-0.03230286765269967):
    eps = tau / steps_md

    def scan_fn(carry, _):
        T_operator = lambda q, p, coef: fast_expi_su3(coef * p) @ q
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
        length=steps_md
    )
        
    return qt, pt

@partial(jax.jit, static_argnums=(2, 4))
def int_MN2_omelyan(q0, p0, F_func, tau, steps_md, lambd=0.1931833275037836):
    eps = tau / steps_md

    def scan_fn(carry, _):
        T_operator = lambda q, p, coef: fast_expi_su3(coef * p) @ q
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
        length=steps_md
    )

    return qt, pt

@partial(jax.jit, static_argnums=(2, 4))
def int_LF2(q0, p0, F_func, tau, steps_md):
    eps = tau / steps_md

    def scan_fn(carry, _):
        T_operator = lambda q, p, coef: fast_expi_su3(coef * p) @ q

        q, p = carry
        p = p - 0.5 * eps * F_func(q)
        q = T_operator(q, p, eps)
        p = p - 0.5 * eps * F_func(q)
        return (q, p), None
    
    (qt, pt), _ = jax.lax.scan(
        f=scan_fn,
        init=(q0, p0),
        length=steps_md
    )

    return qt, pt
