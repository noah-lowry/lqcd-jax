import numpy as np

import torch

def proj_SU3(arr):
    D, V = torch.linalg.eigh(arr.mH @ arr)
    result = arr @ ((V * (torch.unsqueeze(D, dim=-2)**-0.5)) @ V.mH)
    result = result * (torch.unsqueeze(torch.det(result), dim=[-1, -2]) ** (-1/3))
    return result

def unitary_violation(arr, aggregate="mean"):
    """Measure of how much `arr` violates unitarity. Computes |U^H @ U - I|, where || is the Frobenius norm.\n
    `aggregate`: `None, "mean", "max", "rms"`
    """

    I = np.broadcast_to(np.eye(arr.shape[-1], dtype=arr.dtype), arr.shape)

    violation = torch.linalg.matrix_norm(torch.matmul(arr.mH, arr) - I, ord="fro")

    aggregate_fn = {
        None: lambda x: x,
        "mean": torch.mean,
        "max": torch.max,
        "rms": lambda x: torch.sqrt(torch.mean(torch.square(x)))
    }
    
    return aggregate_fn[aggregate](violation)

def special_unitary_grad(func, N=3):
    """Makes the gradient function for a function fn acting on SU(N) elements.\n
    The gradient function returns coefficients for the su(N) generators.
    """
    gradient = torch.grad(func)
    gmm_part = 1j*torch.from_numpy(_make_generators(N)).mH
    return (lambda U, *args, **kwargs: torch.einsum("...ij,nik,...kj->...n", gradient(U, *args, **kwargs), gmm_part.to(device=U.device, dtype=U.dtype), U).real)

def expi_su3(q):
    q = q+0j
    gmm = torch.from_numpy(_make_generators(3)).to(device=q.device, dtype=q.dtype)
    result = expi(torch.einsum("...n,nij->...ij", q, gmm))
    return result

def expi(Q):
    """Compute exp(iQ)."""
    result = torch.matrix_exp(1j*Q)
    return result

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
