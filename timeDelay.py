import numpy as np
import math
from scipy.signal import hilbert
import torch
from utils import *
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
import cmath
import matplotlib.pyplot as plt


# 0 error in unit test
def torchTimeDelay(RP, tau):
    #  Delay a signal in time
    df = RP.Fs / len(RP.transmitSignal)
    #print(fs)
    f_ind = torch.linspace(0, len(RP.transmitSignal) - 1, steps=len(RP.transmitSignal))
    f = f_ind * df
    f[f > (RP.Fs / 2)] -= RP.Fs

    w = (2 * math.pi * f).to(RP.dev)
    arg = w*tau

    if arg.requires_grad == True:
        h = arg.register_hook(lambda x: RP.save(key='arg', val=x))
        RP.hooks.append(h)

    sign = -1.0
    pr = compExp(arg, sign).to(RP.dev)



    #plt.stem(pr.detach().cpu().numpy()[:, 0], use_line_collection=True)
    #plt.show()

    #X = torch.fft(torchHilbert(RP.transmitSignal, RP), 1).to(RP.dev)
    X = torch.rfft(RP.transmitSignal, 1, onesided=False).to(RP.dev)

    #plt.clf()
    #plt.stem(X.detach().cpu().numpy()[:, 0], use_line_collection=True)
    #plt.show()

    #X = torch.fft(torchHilbert(RP.transmitSignal, RP), 1)

    tsd = torch.irfft(compMul(X, pr), 1, onesided=False).to(RP.dev)  # Only return the real values

    return tsd


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
