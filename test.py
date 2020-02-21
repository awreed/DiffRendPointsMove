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
    x = torch.randn(5, 5)
    print(x)
    indices = torch.tensor([[1, 1, 2, 3, 4]]).long()
    y = torch.gather(x, 1, indices.view(-1, 1))

    print(x)

    print(y)