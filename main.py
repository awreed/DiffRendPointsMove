import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.optim
from RenderParameters import RenderParameters
from simulateSASWaveformsPointSource import simulateSASWaveformsPointSource
from Beamformer import *
import pytorch_ssim
import matplotlib.image as mpimg
import time
from VectorDistribution import *
from Complex import *
from geomloss import SamplesLoss
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Where you left, need to calc means of each projector waveform, figure out how to loop over all waveforms
# and update the loss with breaking the graph

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        print("Fix ur cuda loser")
        dev = "cpu"

    device = torch.device(dev)

    # Data structure for ground truth projector waveforms
    RP_GT = RenderParameters()
    RP_GT.generateTransmitSignal()
    RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    ps_GT = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]], requires_grad=True).cuda()
    GT_Loc = torch.linspace(0, 1, int(RP_GT.nSamples)).view(-1, 1).cuda()
    simulateSASWaveformsPointSource(RP_GT, ps_GT)


    # Data structure for estimate projector waveforms
    RP_EST = RenderParameters()
    RP_EST.generateTransmitSignal()
    RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    ps_EST = torch.tensor([[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]], requires_grad=True).cuda()
    EST_Loc = torch.linspace(0, 1, int(RP_EST.nSamples)).view(-1, 1).cuda()

    # Beamformer to create images from projector waveforms
    BF = Beamformer(sceneDimX=[-2, 2], sceneDimY=[-2, 2], sceneDimZ=[0, 0], nPix=[256, 256, 1], dim=2)
    x_vals = torch.unique(BF.pixPos[:, 0]).numel()
    y_vals = torch.unique(BF.pixPos[:, 1]).numel()
    GT = BF.beamformTest(RP_GT).abs()
    GT_XY = GT.view(x_vals, y_vals)
    plt.imshow(GT_XY.detach().cpu().numpy())
    plt.savefig("pics/GT.png")

    loss_val = 10000
    thresh = 10
    optimizer = torch.optim.SGD([ps_EST], lr=.005, momentum=0.000)
    wass_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, diameter=1.0)
    losses = []
    epochs = 500

    for i in range(0, epochs):
        simulateSASWaveformsPointSource(RP_EST, ps_EST)
        for pData_gt, pData_est in zip(RP_GT.projDataArray, RP_EST.projDataArray):
            a = time.time()
            GT_Wfm = pData_gt.wfmRC.abs() / torch.norm(pData_gt.wfmRC.abs(), p=1)
            EST_Wfm = pData_est.wfmRC.abs() / torch.norm(pData_est.wfmRC.abs(), p=1)
            b = time.time()
            print("Norm: " + str(b - a))
            c = time.time()
            loss = wass_loss(EST_Wfm, EST_Loc, GT_Wfm, GT_Loc)
            losses.append(loss)
            d = time.time()
            print("loss: " + str(d - c))
        final_loss = torch.sum(torch.stack(losses))
        final_loss.backward(retain_graph=True)
        print(final_loss)
        print(ps_EST)
        optimizer.step()
        optimizer.zero_grad()
        losses.clear()

        #Beamform estimate image
        EST = BF.beamformTest(RP_EST).abs()
        EST_XY = EST.view(x_vals, y_vals)

        plt.imshow(EST_XY.detach().cpu().numpy())
        plt.savefig("pics/est" + str(i) + ".png")



