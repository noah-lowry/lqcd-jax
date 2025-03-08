import jax

def int_MN4_takaishi_forcrand(q0, p0, F_func, tau, nfev_approx, theta=0.08398315262876693, rho=0.2539785108410595, lambd=0.6822365335719091, mu=-0.03230286765269967):
    eps = tau / (nfev_approx // 5)

    def scan_fn(carry, _):
        q, p = carry

        p = p - theta * eps * F_func(q)
        q = q + rho * eps * p
        p = p - lambd * eps * F_func(q)
        q = q + mu * eps * p
        p = p - (1 - 2 * (lambd + theta)) * 0.5 * eps * F_func(q)
        q = q + (1 - 2 * (mu + rho)) * eps * p
        p = p - (1 - 2 * (lambd + theta)) * 0.5 * eps * F_func(q)
        q = q + mu * eps * p
        p = p - lambd * eps * F_func(q)
        q = q + rho * eps * p
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
        q, p = carry
        
        q = q + lambd * eps * p
        p = p - 0.5 * eps * F_func(q)
        q = q + (1 - 2 * lambd) * eps * p
        p = p - 0.5 * eps * F_func(q)
        q = q + lambd * eps * p

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
        q, p = carry
        p = p - 0.5 * eps * F_func(q)
        q = q + eps * p
        p = p - 0.5 * eps * F_func(q)
        return (q, p), None
    
    (qt, pt), _ = jax.lax.scan(
        f=scan_fn,
        init=(q0, p0),
        length=nfev_approx
    )

    return qt, pt
