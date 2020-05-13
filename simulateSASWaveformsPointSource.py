import torch
import numpy as np
import ProjData
from timeDelay import *
from utils import *
import time
from delaySignal import *
from torch.autograd import *
import time

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def simulateSASWaveformsPointSource(RP, ps, BI=None, gt=False):
    shape = ps.shape
    numScat = list(shape)[0]

    delaySignal = DelaySignal.apply
    delaySignalBatched = DelaySignalBatched.apply

    #print("Num scatterers" + str(numScat))
    RP.projDataArray.clear()
    for index in BI:
        pData = ProjData.ProjData(projPos=RP.projectors[index, :], Fs=RP.Fs, tDur=RP.tDur)

        #a = time.time()
        #sig1 = delaySignal(ps, pData, RP)
        #b = time.time()
        #sig1 = sig1.detach().cpu().numpy()

        #print(b - a)

        #a = time.time()
        pData.wfm = delaySignalBatched(ps, pData, RP)
        #b = time.time()
        #print(b - a)
        #sig2 = sig2.detach().cpu().numpy()

        #print(np.sum(sig1 - sig2))

        RP.projDataArray.append(pData)
