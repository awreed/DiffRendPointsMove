# Settings -> Project Settings -> Project Interpreter -> Gear Icon (show all) -> Select new project -> click tree icon ->
# select lib/site-packages from the AcousticRendererConda project in order to import torch, cv2, numpy, etc...

import torch
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.optim
from utils import *
from RenderParameters import RenderParameters
from ProjData import ProjData
from timeDelay import *
from scipy.signal import hilbert

fs = 100


def genCos(f):
    T = 1 / fs
    tStop = 1
    x = np.arange(0, tStop, T)
    w = 2 * math.pi * f * x
    _y = np.cos(w)
    return x, _y


def genCosFFT(y):
    _Y = np.fft.fft(y)
    _freq = np.fft.fftfreq(_Y.shape[-1])
    _freq = _freq * fs

    return _freq, _Y


if __name__ == '__main__':
    RP = RenderParameters()

    RP.generateTransmitSignal()

    RP.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=1, rStop=1, zStart=.3, zStop=.3)

    pDataNP = ProjData(projPos=RP.projectors[0, :], Fs=RP.Fs, tdur=RP.tDur)
    pDataTorch = ProjData(projPos=RP.projectors[0, :], Fs=RP.Fs, tdur=RP.tDur)

    tauGT = torch.tensor([.005], requires_grad=True)

    pDataTorch.wfm = torchTimeDelay(RP.transmitSignal, torch.tensor(RP.Fs, requires_grad=True, dtype=torch.float64),
                                    tauGT)

    pDataNP.wfm = timeDelay(RP.transmitSignal.detach().numpy(), RP.Fs, tauGT.detach().numpy())

    pDataNP.RC(RP.transmitSignal.detach().numpy())

    pDataTorch.RCTorch(RP.transmitSignal)

    #plt.stem(pDataNP.wfmRC.real)
    #plt.show()
    #print(pDataTorch.wfmRC)
    #print(pDataNP.wfmRC)

    print(np.sum(pDataTorch.wfmRC[:, 0].detach().numpy() - pDataNP.wfmRC.real))
    print(np.sum(pDataTorch.wfmRC[:, 1].detach().numpy() - pDataNP.wfmRC.imag))

    plt.subplot(2, 1, 1)
    f1 = plt.stem(pDataTorch.wfmRC[:, 0].detach().numpy())

    plt.subplot(2, 1, 2)
    f2 = plt.stem(pDataNP.wfmRC.real, linefmt='-')
    plt.show()