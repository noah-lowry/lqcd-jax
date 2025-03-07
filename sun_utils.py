import importlib

import numpy as np
from opt_einsum import contract

class SUGenerator:
    def __init__(self, N, module="numpy", dtype=None, device=None):
        self.N = N
        self.module = importlib.import_module("torch" if module == "torch" else f"jax.numpy" if module == "jax" else module)
        self.matrix_exp_fn = getattr(importlib.import_module("torch.linalg", "matrix_exp")) if module == "torch" \
            else getattr(importlib.import_module("jax.scipy.linalg" if module == "jax" else "scipy.linalg"), "expm")
        self.dtype = dtype
        self.device = device

        if module not in ("numpy", "torch", "jax"):
            raise ValueError("module must be numpy or torch or jax.")
        
        self.generators = self._make_generators(N)

        if module == "torch":
            self.generators = module.from_numpy(self.generators)
            if device is not None:
                self.generators = self.generators.to(device=device)
        if module == "jax":
            self.generators = self.module.array(self.generators)

        if module != "torch" and device is not None:
            raise ValueError("device can only be used with torch, set to None using numpy or jax.")
        if dtype is not None:
            if (module == "torch" and not dtype.is_complex) or not self.module.iscomplexobj(dtype(0)):
                raise TypeError("dtype must be a complex type.")
            self.generators = self._to_dtype(self.generators, dtype)
    
    def __call__(self, coef):
        return self.generate_group(coef)
    
    def generate_algebra(self, coef):
        # if self.module == "torch":
        #     assert coef.device == self.generators.device
        # assert coef.ndim >= 1
        # assert coef.shape[-1] == self.N**2 - 1

        coef = self._to_dtype(coef, self.generators.dtype)
        su_N = contract("...N,Nij->...ij", coef, self.generators)
        return su_N
    
    def _to_dtype(self, arr, dtype):
        return arr.to(dtype=dtype) if self.module == "torch" else arr.astype(dtype)
    
    def generate_group(self, coef):
        su_N = self.generate_algebra(coef)
        SU_N = self.matrix_exp_fn(1j*su_N)
        return SU_N
        
    def _alpha_index(self, n, m):
        assert m < n
        return n*n + 2*(m-n) - 2

    def _beta_index(self, n, m):
        assert m < n
        return n*n + 2*(m-n) - 1

    def _gamma_index(self, n):
        return n*n - 2

    def _make_generators(self, N):
        result = np.empty((N*N-1, N, N), dtype=np.complex128)
        for m in range(1, N+1):
            for n in range(m+1, N+1):
                gen = np.zeros((N, N), dtype=np.complex128)
                gen[m-1, n-1] = gen[n-1, m-1] = 0.5
                result[self._alpha_index(n, m)] = gen

                gen = np.zeros((N, N), dtype=np.complex128)
                gen[m-1, n-1] = -0.5j
                gen[n-1, m-1] = 0.5j
                result[self._beta_index(n, m)] = gen

                gen = np.diag(np.array([1,]*(n-1)+[0,]*(N+1-n), dtype=np.complex128))
                gen[n-1, n-1] = 1-n
                gen /= np.sqrt(2*n*(n-1))
                result[self._gamma_index(n)] = gen
        return result