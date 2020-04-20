import torch
from RenderParameters import *
from simulateSASWaveformsPointSource import *
from utils import *
from pytorch_complex_tensor import ComplexTensor
import matplotlib.pyplot as plt

def simulateWaveformsBatched(RP, ps, BI=None, propLoss=False):
    shape = ps.shape
    numScat = list(shape)[0]
    RP.projDataArray = []
    atten_window = (torch.linspace(1, 0, RP.nSamples)**2).to(RP.dev)
    if BI is None:
        for i in range(0, RP.numProj):
            pData = ProjData.ProjData(projPos=RP.projectors[i, :], Fs=RP.Fs, tDur=RP.tDur)
            dist = torch.sqrt(torch.sum((pData.projPos.repeat(numScat, 1) - ps[:, :])**2, 1))
            tau = (dist * 2) / RP.c - RP.minDist/RP.c
            if propLoss is True:
                wfms = timeDelayBatched(RP, tau)*atten_window
            else:
                wfms = timeDelayBatched(RP, tau)

            pData.wfm = wfms
            pData.RCTorch(RP)
            #plt.stem(pData.wfmRC.abs().detach().cpu().numpy(), use_line_collection=True)
            #plt.show()
            RP.projDataArray.append(pData)
    else:
        for index in BI:
            #print(index)
            pData = ProjData.ProjData(projPos=RP.projectors[index, :], Fs=RP.Fs, tDur=RP.tDur)
            dist = torch.sqrt(torch.sum((pData.projPos.repeat(numScat, 1) - ps[:, :])**2, 1))
            #if  ps.requires_grad == True:
            #    dist.register_hook(lambda x: print("dist gradient" + str(x)))
            #    RP.hooks.append(dist)
            tau = (dist * 2) / RP.c - RP.minDist/RP.c


            #if  ps.requires_grad == True:
            #    tau.register_hook(lambda x: print("tau gradient" + str(x)))
            #    RP.hooks.append(tau)

            if propLoss is True:
                wfms = timeDelayBatched(RP, tau) * atten_window
            else:
                wfms = timeDelayBatched(RP, tau)
            pData.wfm = wfms
            pData.RCTorch(RP)
            RP.projDataArray.append(pData)

def compExp(w, sign):
    real = torch.cos(w)
    imag = sign * torch.sin(w)
    return torch.stack((real, imag), 2)

def timeDelayBatched(RP, tau):
    ValueError(tau.dim() > 1, "Tau should be 1-d vector or scalar element")
    df = RP.Fs / (len(RP.transmitSignal) * 1.0)
    f_ind = torch.linspace(0, len(RP.transmitSignal) - 1, steps=len(RP.transmitSignal), dtype=torch.float64)
    f = f_ind * df
    f[f > (RP.Fs / 2)] -= RP.Fs
    w = (2 * math.pi * f).to(RP.dev)
    arg = torch.mm(tau.unsqueeze(1), w.unsqueeze(0))
    #arg = 2*math.pi*torch.mm(tau.unsqueeze(1), f.unsqueeze(0))  # tau = [t1, t2, t3] * f vector
    sign = -1.0

    pr = compExp(arg, sign).to(RP.dev)

    # Complex multiply that works for batches. Should use GPU
    ac = torch.mul(RP.Pulse[:, 0], pr[:, :, 0])
    bd = torch.mul(RP.Pulse[:, 1], pr[:, :, 1])
    bc = torch.mul(RP.Pulse[:, 1], pr[:, :, 0])
    ad = torch.mul(RP.Pulse[:, 0], pr[:, :, 1])
    mul = torch.stack((ac - bd, bc + ad), 2)
    tsd = torch.ifft(mul, 1)[:, :, 0]
    return torch.sum(tsd, 0)

if __name__ == "__main__":
    RP = RenderParameters()
    RP.generateTransmitSignal()
    RP.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    ps = torch.tensor([[-1.0, .5], [.25, .5], [2.0, 1.0], [1.5, -1.0]], requires_grad=True).cuda()

    simulateWaveformsBatched(RP, ps)
