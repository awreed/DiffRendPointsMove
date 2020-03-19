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
import random
import collections
from batch_time_delay import *
import visdom

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Where you left, need to calc means of each projector waveform, figure out how to loop over all waveforms
# and update the loss with breaking the graph

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
        dev_1 = "cuda:1"
        torch.cuda.empty_cache()
    else:
        print("Fix ur cuda loser")
        dev = "cpu"

    #device = torch.device(dev)
    BS = 100
    with torch.no_grad():
        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        RP_GT.generateTransmitSignal()
        RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=10, rStart=1, rStop=1, zStart=.3, zStop=.3)
        ps_GT = torch.tensor([[-.3, -.3], [.3, -.3]], requires_grad=False).to(dev)
        simulateSASWaveformsPointSource(RP_GT, ps_GT)
        print(RP_GT.numProj)

        # Beamformer to create images from projector wavef orms
        BF_GT = Beamformer(RP=RP_GT, sceneDimX=[-.4, .4], sceneDimY=[-.4, .4], sceneDimZ=[0, 0], nPix=[128, 128, 1], dim=2)
        x_vals = torch.unique(BF_GT.pixPos[:, 0]).numel()
        y_vals = torch.unique(BF_GT.pixPos[:, 1]).numel()
        GT = BF_GT.softBeamformer(RP_GT).abs().to(dev_1)
        GT_XY = GT.view(x_vals, y_vals)
        plt.imshow(GT_XY.detach().cpu().numpy())
        plt.savefig("pics1/GT.png")

        RP_EST = RenderParameters(device=dev_1)
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=10, rStart=2, rStop=2, zStart=.3, zStop=.3)

        BF_EST = Beamformer(RP=RP_EST, sceneDimX=[-.4, .4], sceneDimY=[-.4, .4], sceneDimZ=[0, 0], nPix=[128, 128, 1], dim=2)

        ps_EST = torch.tensor([[.3, .3], [-.3, .3]], requires_grad=False).to(dev_1)

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='Wass-1 Loss', legend=['Loss']))

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    optimizer = torch.optim.Adam([ps_est], lr=0.01)

    epochs = 1000

    for i in range(0, epochs):
        simulateSASWaveformsPointSource(RP_EST, ps_est)

        est = BF_EST.softBeamformer(RP_EST).abs()
        #est.register_hook(lambda x: print("est " + str(x.data)))

        est_xy = est.view(x_vals, y_vals)
        plt.imshow(est_xy.detach().cpu().numpy())
        #plt.savefig("pics1/est" + str(i) + ".png")
        plt.pause(.05)

        loss = torch.sum(torch.sqrt((GT - est)**2))

        print(ps_est)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window, update='append')
