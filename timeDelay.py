import numpy as np
import math
from scipy.signal import hilbert
import torch
from utils import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import cmath
import matplotlib.pyplot as plt


# 0 error in unit test
def torchTimeDelay(x, fs, tau, RP):
    #  Delay a signal in time
    df = fs / torch.tensor([len(x) * 1.0], requires_grad=False)
    f_ind = torch.linspace(0, len(x) - 1, steps=len(x))
    f = f_ind * df.item()
    f[f > (fs.item() / 2)] -= fs.item()

    arg = 2*torch.tensor([math.pi])*tau*f
    #h = arg.register_hook(lambda z: print(torch.sum(z)))
    #RP.hooks.append(h)

    sign = torch.tensor([-1.0], requires_grad=True)
    pr = compExp(arg, sign)
    X = torch.fft(torchHilbert(x), 1)
    tsd = torch.ifft(compMul(X, pr), 1)[:, 0]  # Only return the real values

    return tsd.cuda()


def timeDelay(x, fs, tau):
    #  Delay a signal in time
    df = fs / len(x)
    f = np.linspace(0, len(x) - 1, len(x)) * df
    f[f > (fs / 2)] = f[f > (fs / 2)] - fs
    arg = 2 * math.pi*tau*f
    pr = np.exp(-1j * 2 * math.pi * tau * f)
    #print(pr)

    #print(pr)

    if np.isreal(x).any():
        X = np.fft.fft(hilbert(x))
        tsd = (np.fft.ifft(np.multiply(X, pr))).real
    else:
        X = np.fft(x)
        tsd = np.fft.ifft(np.multiply(X, np.conj(pr)))
    return tsd
