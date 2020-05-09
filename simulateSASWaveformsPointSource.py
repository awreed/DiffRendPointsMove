import torch
import numpy as np
import ProjData
from timeDelay import *
from utils import *
import time
from delaySignal import *
from torch.autograd import *


# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def simulateSASWaveformsPointSource(RP, ps, BI=None, gt=False):
    shape = ps.shape
    numScat = list(shape)[0]

    delaySignal = DelaySignal.apply

    #print("Num scatterers" + str(numScat))
    RP.projDataArray.clear()
    for index in BI:
        pData = ProjData.ProjData(projPos=RP.projectors[index, :], Fs=RP.Fs, tDur=RP.tDur)

        pData.wfm = delaySignal(ps, pData, RP)

        #if ps.requires_grad:
        #    h = ps.register_hook(lambda x: print(x))
        #    RP.hooks.append(h)

           # sig = torch.zeros(RP.nSamples)
           # start = (tau * RP.Fs).long()
           ## sig[start:start+5] = RP.transmitSignal
            #if ps.requires_grad == True:
            ##    sig.requires_grad = True
             #   h = sig.register_hook(lambda x: RP.save(key='indSig', val=sig))
             #   RP.hooks.append(h)

        #if ps.requires_grad == True:
        #    h = pData.wfm.register_hook(lambda x: RP.save(key='pDataWfm', val=x))
        #    RP.hooks.append(h)

        #pData.wfm = (pData.wfm - pData.wfm.min().detach())/(pData.wfm.max().detach() - pData.wfm.min().detach())
        #print("here")
        #plt.clf()
        #plt.stem(pData.wfm.detach().cpu().numpy(), use_line_collection=True)
        #plt.show()
        # print(pData.wfm.shape)
        #pData.RCTorch(RP)
        RP.projDataArray.append(pData)
