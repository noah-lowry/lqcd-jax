from functools import partial

import torch

from special_unitary import FastExpiSU3

# Integrators here are from https://arxiv.org/pdf/hep-lat/0505020

def _torch_scan(f, init, length):
    val = init
    ys = []
    for x in range(length):
        val, y = f(val, x)
        ys.append(y)
    ys = torch.stack(ys)
    return val, ys

def int_2LF(q0, p0, F_func, tau, steps_md):
    eps = tau / steps_md

    T_operator = lambda q, p, coef: torch.matmul(FastExpiSU3.apply(coef * eps * p), q)
    V_operator = lambda q, p, coef: p - (coef * eps * F_func(q))

    def scan_fn(carry, _):
        q, p = carry

        p = V_operator(q, p, 0.5)
        q = T_operator(q, p, eps)
        p = V_operator(q, p, 0.5)

        return (q, p), None
    
    (qt, pt), _ = _torch_scan(
        f=scan_fn,
        init=(q0, p0),
        length=steps_md
    )

    return qt, pt

def int_2MN(q0, p0, F_func, tau, steps_md, lambd=0.1931833275037836):
    eps = tau / steps_md

    T_operator = lambda q, p, coef: torch.matmul(FastExpiSU3.apply(coef * eps * p), q)
    V_operator = lambda q, p, coef: p - (coef * eps * F_func(q))

    def scan_fn(carry, _):
        q, p = carry
        
        q = T_operator(q, p, lambd)
        p = V_operator(q, p, 0.5)
        q = T_operator(q, p, 1 - 2*lambd)
        p = V_operator(q, p, 0.5)
        q = T_operator(q, p, lambd)

        return (q, p), None
    
    (qt, pt), _ = _torch_scan(
        f=scan_fn,
        init=(q0, p0),
        length=steps_md
    )

    return qt, pt

def int_4MN4FP(q0, p0, F_func, tau, steps_md, rho=0.1786178958448091, theta=-0.06626458266981843, lambd=0.7123418310626056):
    eps = tau / steps_md

    T_operator = lambda q, p, coef: torch.matmul(FastExpiSU3.apply(coef * eps * p), q)
    V_operator = lambda q, p, coef: p - (coef * eps * F_func(q))

    def scan_fn(carry, _):
        q, p = carry

        q = T_operator(q, p, rho)
        p = V_operator(q, p, lambd)
        q = T_operator(q, p, theta)
        p = V_operator(q, p, (1 - 2*lambd) / 2)
        q = T_operator(q, p, 1 - 2*(theta + rho))
        p = V_operator(q, p, (1 - 2*lambd) / 2)
        q = T_operator(q, p, theta)
        p = V_operator(q, p, lambd)
        q = T_operator(q, p, rho)
        
        return (q, p), None
    
    (qt, pt), _ = _torch_scan(
        f=scan_fn,
        init=(q0, p0),
        length=steps_md
    )
        
    return qt, pt

def int_4MN5FV(q0, p0, F_func, tau, steps_md, theta=0.08398315262876693, rho=0.2539785108410595, lambd=0.6822365335719091, mu=-0.03230286765269967):
    eps = tau / steps_md

    T_operator = lambda q, p, coef: torch.matmul(FastExpiSU3.apply(coef * eps * p), q)
    V_operator = lambda q, p, coef: p - (coef * eps * F_func(q))

    def scan_fn(carry, _):
        q, p = carry

        p = V_operator(q, p, theta)
        q = T_operator(q, p, rho)
        p = V_operator(q, p, lambd)
        q = T_operator(q, p, mu)
        p = V_operator(q, p, (1 - 2*(lambd + theta)) / 2)
        q = T_operator(q, p, 1 - 2*(mu + rho))
        p = V_operator(q, p, (1 - 2*(lambd + theta)) / 2)
        q = T_operator(q, p, mu)
        p = V_operator(q, p, lambd)
        q = T_operator(q, p, rho)
        p = V_operator(q, p, theta)

        return (q, p), None
    
    (qt, pt), _ = _torch_scan(
        f=scan_fn,
        init=(q0, p0),
        length=steps_md
    )
        
    return qt, pt
