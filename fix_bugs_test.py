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
from Wavefront import *
import pickle
from utils import *

# When emailing Tom:
# Object size is important
# Smaller scene saves memory if I range gate

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

    BS = 120
    thetaStart = 0
    thetaStop = 359
    thetaStep = 1
    rStart = 1
    rStop = 1
    zStart = 0.3
    zStop = 0.3
    sceneDimX = [-.4, .4]
    sceneDimY = [-.4, .4]
    sceneDimZ = [-.4, .4]
    pixDim = [128, 128, 32]
    propLoss = True

    with torch.no_grad():
        ps_GT = torch.tensor([[-.3, -.3, 0.0]], requires_grad=False).to(dev)
        ps_EST = torch.tensor([[.3, .3, 0.0]], requires_grad=False).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep,
                                 rStart=rStart, rStop=rStop, zStart=zStart, zStop=zStop)
        RP_GT.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_GT.generateTransmitSignal()

        simulateWaveformsBatched(RP_GT, ps_GT, propLoss=propLoss)

        RP_EST = RenderParameters(device=dev_1)
        # RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop)
        RP_EST.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_EST.generateTransmitSignal()

        BF_EST = Beamformer(RP=RP_EST)

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='fzsdfepoch', ylabel='Loss', title='L1 Loss', legend=['Loss']))
        wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    optimizer = torch.optim.Adam([ps_est], lr=.01)

    epochs = 100000
    Projs = range(0, RP_EST.numProj)
    batches = list(batchFromList(Projs, BS))
    losses = []

    for i in range


