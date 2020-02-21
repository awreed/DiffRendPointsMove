import torch
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def DRC_Isaac_Vectorized(img, med, des_med):
    free_parameter = (des_med - med * des_med)/(med - med * des_med)
    return (img * free_parameter)/(free_parameter * img - img + 1)


def compABS(t):
    r = t[:, 0]
    i = t[:, 1]
    return torch.sqrt((r**2 + i**2))

def compConj(t1):
    r = t1[:, 0]
    imag = -1 * t1[:, 1]
    return torch.stack((r, imag), 1)


def compExp(w, sign):
    # Returns N X 2 array of complex exp where e^(jw) = cos(w) + sign*j*sin(w)
    # w is the input array to the exponential
    # sign should be -1 or 1 to indicate e^(-jw) -r e^(jw) respectively
    ValueError(abs(sign) == 1, 'Complex exponential sign should be -1 or 1')
    #val = torch.zeros((len(w), 2), requires_grad=True, dtype=torch.float64)
    real = torch.cos(w)
    imag = sign * torch.sin(w)
    return torch.stack((real, imag), 1)


def compMul(t1, t2):
    # Multiply two complex numbers together
    # I should really make a class for this
    a = t1[:,0]
    b = t1[:,1]
    c = t2[:,0]
    d = t2[:,1]
    val = torch.stack((a*c-b*d, b*c+a*d),1)
    return val


def cvtNP(t):
    if isinstance(t, torch.Tensor):
        return t.detach().numpy()
    else:
        return t

def xcorr(t1, t2):
    #t1_hil = torchHilbert(t1)
    #t1_norm = t1/torch.norm(t1, p=1.0)
    T1 = torch.rfft(t1, 1, onesided=False)

    #t2_hil = torchHilbert(t2)
    #t2_norm = t2/torch.norm(t2, p=1.0)
    T2 = torch.rfft(t2, 1, onesided=False)

    RC = torch.ifft(compMul(T2, compConj(T1)), 1)[:, 0]

    return RC





# https://stackoverflow.com/questions/56380536/hilbert-transform-in-python
# 0 error against Scipy and works with autograd
def torchHilbert(u):
    N = len(u)

    # Take forward fourier transform
    U = torch.rfft(u, 1, onesided=False)
    Mask = torch.zeros(N, requires_grad=True)

    if N % 2 == 0:
        DC = torch.tensor([1], requires_grad=True, dtype=torch.float64)
        DC1 = torch.tensor([1], requires_grad=True, dtype=torch.float64)
        maskLeft = torch.ones((N // 2 - 1), requires_grad=True, dtype=torch.float64)
        maskLeftUp = maskLeft * 2
        maskRight = torch.zeros((N - (N // 2 - 1) - 2), requires_grad=True, dtype=torch.float64)
        Mask = torch.cat((DC, maskLeftUp, DC1, maskRight), 0)
    else:
        DC = torch.tensor([1], requires_grad=True, dtype=torch.float64)
        maskLeft = torch.ones(((N + 1) // 2 - 1), requires_grad=True, dtype=torch.float64)
        maskLeftUp = maskLeft * 2
        maskRight = torch.zeros((N - ((N + 1) // 2 - 1) - 1), requires_grad=True, dtype=torch.float64)
        Mask = torch.cat((DC, maskLeftUp, maskRight), 0)

    # Zero out negative frequency components

    real = torch.mul(U[:, 0], Mask)
    imag = torch.mul(U[:, 1], Mask)
    U_Pos = torch.stack((real, imag), 1)

    # Take inverse Fourier transform
    v = torch.ifft(U_Pos, 1)
    return v
