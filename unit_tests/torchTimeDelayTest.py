# Settings -> Project Settings -> Project Interpreter -> Gear Icon (show all) -> Select new project -> click tree icon ->
# select lib/site-packages from the AcousticRendererConda project in order to import torch, cv2, numpy, etc...

import torch
from torch import *
import torch.distributions
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.optim
from utils import *
from RenderParameters import RenderParameters
from ProjData import ProjData
from timeDelay import *
from VectorDistribution import *
from scipy.signal import hilbert
torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

    pData = ProjData(projPos=RP.projectors[0, :], Fs=RP.Fs, tdur=RP.tDur)

    tauGT = torch.tensor([.005], requires_grad=True).cuda()
    tauEST = torch.tensor([0.009], requires_grad=True).cuda()

    sigGT = torchTimeDelay(RP.transmitSignal, torch.tensor(RP.Fs, requires_grad=True),
                           tauGT)

    distSigGT = vectorDistribution(probs=sigGT)
    print(distSigGT.mean)

    #print(distSigGT.mean)

    criterion = torch.nn.PairwiseDistance(p=1.0)
    learning_rate = 1e-10

    loss = 100

    optimizer = torch.optim.SGD([tauEST], lr=.00000000001)
    #a = .4
    #b = 1 - a

    while abs(loss) > 0.00001:
        optimizer.zero_grad()
        #tauESTClamp = torch.clamp(tauEST, 0, 0.02)
        sigEst = torchTimeDelay(RP.transmitSignal, torch.tensor(RP.Fs, requires_grad=True),
                                tauEST)
        distSigEst = vectorDistribution(probs=sigEst)
        mean_loss = (distSigGT.mean - distSigEst.mean)**2
        L1_loss = .0001*criterion(sigEst.unsqueeze(0), sigGT.unsqueeze(0)) + mean_loss
        #print(mean_loss)
        #print(L1_loss)
        loss = mean_loss + L1_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        #print(tauESTClamp)

        plt.cla()
        plt.plot(sigGT.clone().detach().cpu().numpy(), color="blue")
        plt.plot(sigEst.clone().detach().cpu().numpy(), color="red")
        plt.pause(0.05)
    plt.show()
