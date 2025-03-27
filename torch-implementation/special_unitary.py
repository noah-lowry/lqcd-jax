import numpy as np

import torch

def proj_SU3(arr):
    D, V = torch.linalg.eigh(arr.clone().mH @ arr)
    result = arr @ ((V * (torch.unsqueeze(D, dim=-2)**-0.5)) @ V.mH)
    result = result * (torch.det(result)[..., None, None] ** (-1/3))
    return result

def unitary_violation(arr):
    """Measure of how much `arr` violates unitarity. Computes |U^H @ U - I|, where || is the Frobenius norm.\n
    """

    I = torch.broadcast_to(torch.eye(arr.shape[-1], dtype=arr.dtype, device=arr.device), arr.shape)

    violation = torch.linalg.matrix_norm(torch.matmul(arr.mH, arr) - I, ord="fro")

    return violation.mean()

def special_unitary_grad(func):
    """Makes the gradient function for a function fn acting on SU(3) elements.\n
    The gradient function returns coefficients for the su(3) generators.
    """
    # gradient = torch.func.grad(func)
    gmm_part = 1j*torch.tensor([[[ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j],
        [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j, -0.0000-0.5000j,  0.0000+0.0000j],
        [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j, -0.5000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
        [ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
        [ 0.0000+0.0000j,  0.0000+0.5000j,  0.0000+0.0000j]],

        [[ 0.2886751345948129+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.2886751345948129+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j, -0.5773502691896258+0.0000j]]]).mH

    return (lambda U: torch.func.grad(lambda w: func(expi(w) @ U))(torch.zeros(*U.shape[:-2], 8, dtype=U.dtype, device=U.device)).real)
    # return (lambda U, *args, **kwargs: torch.einsum("...ij,nik,...kj->...n", gradient(U, *args, **kwargs), gmm_part.to(device=U.device, dtype=U.dtype), U).real)

def expi(q):
    q = q + 0j
    gmm = torch.tensor([[[ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j],
        [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j, -0.0000-0.5000j,  0.0000+0.0000j],
        [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j, -0.5000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
        [ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j]],

        [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
        [ 0.0000+0.0000j,  0.0000+0.5000j,  0.0000+0.0000j]],

        [[ 0.2886751345948129+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.2886751345948129+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.0000j, -0.5773502691896258+0.0000j]]], dtype=q.dtype, device=q.device)

    Q = torch.einsum("...N,Nij->...ij", q, gmm)
    return torch.linalg.matrix_exp(1j*Q)


# class FastExpiSU3(torch.autograd.Function):
    
#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         q, = inputs
#         ctx.save_for_backward(q)
    
#     @staticmethod
#     def forward(q):
#         q = q + 0j
#         gmm = torch.tensor([[[ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j],
#             [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j, -0.0000-0.5000j,  0.0000+0.0000j],
#             [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j, -0.5000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
#             [ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
#             [ 0.0000+0.0000j,  0.0000+0.5000j,  0.0000+0.0000j]],

#             [[ 0.2886751345948129+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.2886751345948129+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j, -0.5773502691896258+0.0000j]]], dtype=q.dtype, device=q.device)

#         Q = torch.einsum("...N,Nij->...ij", q, gmm)

#         Q__2 = torch.matmul(Q, Q)

#         c0 = torch.einsum("...AB,...BA->...", Q__2, Q).real / 3
#         c1_3 = torch.einsum("...ii->...", Q__2.real) / 6
#         c0_max = torch.pow(c1_3, 1.5) * 2
        
#         where_close_to_zero = torch.isclose(c0_max, torch.zeros_like(c0_max))
#         ones = torch.ones_like(c0_max)

#         theta = torch.acos(
#             torch.where(
#                 where_close_to_zero, ones, torch.abs(c0) / c0_max
#             )  # Tr(Q^2) == 0 -> Tr(Q^3) == 0 (additionally, Tr(Q^2) == 0 -> Q == 0)
#         )
#         u__2 = c1_3 * torch.cos(theta / 3)**2
#         w__2 = (c1_3 - u__2) * 3

#         u = torch.sqrt(u__2)
#         w = torch.sqrt(w__2)

#         cos_w = torch.cos(w)
#         sinc_w = torch.sinc(w / torch.pi)
#         exp_miu = torch.exp(-1j*u)
#         exp_2iu = 1 / (exp_miu * exp_miu)

#         h0 = (u__2 - w__2) * exp_2iu + exp_miu * (8*u__2*cos_w + 2j*u*(3*u__2 + w__2)*sinc_w)
#         h1 = 2*u*exp_2iu - exp_miu * (2*u*cos_w - 1j*(3*u__2 - w__2)*sinc_w)
#         h2 = exp_2iu - exp_miu * (cos_w + 3j*u*sinc_w)

#         dd = 9*u__2 - w__2

#         f0 = torch.where(where_close_to_zero, ones, h0 / dd)
#         f1 = torch.where(where_close_to_zero, 1j*ones, h1 / dd)
#         f2 = torch.where(where_close_to_zero, ones, h2 / dd)
        
#         c0_sign = torch.signbit(c0)
#         f0 = torch.where(c0_sign, f0.conj(), f0)
#         f1 = torch.where(c0_sign, -f1.conj(), f1)
#         f2 = torch.where(c0_sign, f2.conj(), f2)

#         I = torch.broadcast_to(torch.eye(3, dtype=Q.dtype, device=Q.device), Q.shape)
#         expi = f0[..., None, None]*I + f1[..., None, None]*Q + f2[..., None, None]*Q__2

#         return expi
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         q, = ctx.saved_tensors
#         q = q + 0j
#         gmm = torch.tensor([[[ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j],
#             [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j, -0.0000-0.5000j,  0.0000+0.0000j],
#             [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j, -0.5000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.5000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.5000j,  0.0000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.5000+0.0000j],
#             [ 0.0000+0.0000j,  0.5000+0.0000j,  0.0000+0.0000j]],

#             [[ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j, -0.0000-0.5000j],
#             [ 0.0000+0.0000j,  0.0000+0.5000j,  0.0000+0.0000j]],

#             [[ 0.2886751345948129+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.2886751345948129+0.0000j,  0.0000+0.0000j],
#             [ 0.0000+0.0000j,  0.0000+0.0000j, -0.5773502691896258+0.0000j]]], dtype=q.dtype, device=q.device)

#         Q = torch.einsum("...N,Nij->...ij", q, gmm)

#         Q__2 = torch.matmul(Q, Q)
#         c0 = torch.einsum("...AB,...BA->...", Q__2, Q).real / 3
#         c1_3 = torch.einsum("...ii->...", Q__2.real) / 6
#         c0_max = torch.pow(c1_3, 1.5) * 2

#         where_close_to_zero = torch.isclose(c0_max, torch.zeros_like(c0_max))
#         ones = torch.ones_like(c0_max)

#         theta = torch.acos(
#             torch.where(
#                 where_close_to_zero, ones, torch.abs(c0) / c0_max
#             )  # Tr(Q^2) == 0 -> Tr(Q^3) == 0 (additionally, Tr(Q^2) == 0 -> Q == 0)
#         )
#         u__2 = c1_3 * torch.cos(theta / 3)**2
#         w__2 = (c1_3 - u__2) * 3

#         u = torch.sqrt(u__2)
#         w = torch.sqrt(w__2)

#         cos_w = torch.cos(w)
#         sinc_w = torch.sinc(w / torch.pi)
#         exp_miu = torch.exp(-1j*u)
#         exp_2iu = 1 / (exp_miu * exp_miu)

#         h0 = (u__2 - w__2) * exp_2iu + exp_miu * (8*u__2*cos_w + 2j*u*(3*u__2 + w__2)*sinc_w)
#         h1 = 2*u*exp_2iu - exp_miu * (2*u*cos_w - 1j*(3*u__2 - w__2)*sinc_w)
#         h2 = exp_2iu - exp_miu * (cos_w + 3j*u*sinc_w)

#         dd = 9*u__2 - w__2

#         f0 = torch.where(where_close_to_zero, ones, h0 / dd)
#         f1 = torch.where(where_close_to_zero, 1j*ones, h1 / dd)
#         f2 = torch.where(where_close_to_zero, ones, h2 / dd)

#         xi1 = (cos_w - sinc_w) / w__2

#         r01 = 2*(u+1j*(u__2-w__2))*exp_2iu + 2*exp_miu*(4*u*(2-1j*u)*cos_w + 1j*(9*u__2 + w__2 - 1j*u*(3*u__2+w__2))*sinc_w)
#         r11 = 2*(1+2j*u)*exp_2iu + exp_miu*(-2*(1-1j*u)*cos_w + 1j*(6*u+1j*(w__2-3*u__2))*sinc_w)
#         r21 = 2j*exp_2iu + 1j*exp_miu*(cos_w - 3*(1-1j*u)*sinc_w)
#         r02 = -2*exp_2iu + 2j*u*exp_miu*(cos_w + (1+4j*u)*sinc_w + 3*u__2*xi1)
#         r12 = -1j*exp_miu*(cos_w + (1+2j*u)*sinc_w - 3*u__2*xi1)
#         r22 = exp_miu*(sinc_w - 3j*u*xi1)

#         ddd = 2*dd**2

#         c0_sign = torch.signbit(c0)
#         b10 = (2*u*r01 + (3*u__2 - w__2)*r02 - 2*(15*u__2 + w__2)*f0)
#         b10 = torch.where(where_close_to_zero, ones, b10 / ddd)
#         b10 = torch.where(c0_sign, torch.conj(b10), b10)

#         b11 = (2*u*r11 + (3*u__2 - w__2)*r12 - 2*(15*u__2 + w__2)*f1)
#         b11 = torch.where(where_close_to_zero, ones, b11 / ddd)
#         b11 = torch.where(c0_sign, -torch.conj(b11), b11)

#         b12 = (2*u*r21 + (3*u__2 - w__2)*r22 - 2*(15*u__2 + w__2)*f2)
#         b12 = torch.where(where_close_to_zero, ones, b12 / ddd)
#         b12 = torch.where(c0_sign, torch.conj(b12), b12)

#         b20 = (r01 - 3*u*r02 - 24*u*f0)
#         b20 = torch.where(where_close_to_zero, ones, b20 / ddd)
#         b20 = torch.where(c0_sign, -torch.conj(b20), b20)

#         b21 = (r11 - 3*u*r12 - 24*u*f1)
#         b21 = torch.where(where_close_to_zero, ones, b21 / ddd)
#         b21 = torch.where(c0_sign, torch.conj(b21), b21)

#         b22 = (r21 - 3*u*r22 - 24*u*f2)
#         b22 = torch.where(where_close_to_zero, ones, b22 / ddd)
#         b22 = torch.where(c0_sign, -torch.conj(b22), b22)

#         f0 = torch.where(c0_sign, torch.conj(f0), f0)
#         f1 = torch.where(c0_sign, -torch.conj(f1), f1)
#         f2 = torch.where(c0_sign, torch.conj(f2), f2)

#         I = torch.broadcast_to(torch.eye(3, dtype=Q.dtype, device=Q.device), Q.shape)
#         B1 = b10[..., None, None]*I + b11[..., None, None]*Q + b12[..., None, None]*Q__2
#         B2 = b20[..., None, None]*I + b21[..., None, None]*Q + b22[..., None, None]*Q__2
        

#         QT = torch.einsum("...ik,Nkj->...ijN", Q, gmm)
#         TQ = torch.einsum("Nik,...kj->...ijN", gmm, Q)

#         Tr1 = torch.einsum("...iiN->...N", QT)
#         Tr2 = torch.einsum("...ij,Nji->...N", Q__2, gmm)
        
#         grad_input = torch.conj(torch.einsum("...ij,...N,...ij->...N", grad_output, Tr1, B1) + \
#                                 torch.einsum("...ij,...N,...ij->...N", grad_output, Tr2, B2) + \
#                                 torch.einsum("...ij,...,Nij->...N", grad_output, f1, gmm) + \
#                                 torch.einsum("...ij,...,...ijN->...N", grad_output, f2, TQ+QT))
        
#         return grad_input
