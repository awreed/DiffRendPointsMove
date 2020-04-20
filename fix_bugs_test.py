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
import torch.nn.functional as F


#idea - similar to loss landspace
# create several points with several zs and a single ring of projectors
# display the beamformed images of the points at different z values and see if the ring changes much at all


# When emailing Tom:
# Object size is important
# Smaller scene saves memory if I range gate

#Comes down to projector smapling


# Where you left, need to calc means of each projector waveform, figure out how to loop over all waveforms
# and update the loss with breaking the graph
np.random.seed(0)
random.seed(0)

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
        dev_1 = "cuda:1"
        torch.cuda.empty_cache()
    else:
        print("Fix ur cuda loser")
        dev = "cpu"

    BS = 50
    #xStart = -1
    #xStop = 1
    #yStart = -1
    #yStop = 1
    #zStart = 0.3
    #zStop = 0.3
    #xStep = .01
    #yStep = .01

    thetaStart = 0
    thetaStop = 359
    thetaStep = 1
    rStart = 1
    rStop = 1
    zStart = 1
    zStop = 1
    zStep = 0.1
    sceneDimX = [-.4, .4] # min dist needs to equal max dist
    sceneDimY = [-.4, .4]
    sceneDimZ = [-.4, .4]
    pixDim = [128, 128, 128]
    propLoss = True
    compute = True
    show=False

    with torch.no_grad():
        ps_GT = torch.tensor([[-.3, -.3, -.3]], requires_grad=False).to(dev)
        ps_EST = torch.tensor([[.3, 0.3, .3]], requires_grad=False).to(dev_1)

        #ps_GT = torch.tensor([[0, -.3, -.3]], requires_grad=False).to(dev)
        #ps_EST = torch.tensor([[0, 0.3, .3]], requires_grad=False).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        #RP_GT.defineProjectorPosSpiral(show=False)
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop, zStep=zStep)
        RP_GT.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_GT.generateTransmitSignal(compute=compute)
        print("Number of projectors")
        print(RP_GT.numProj)

        #simulateWaveformsBatched(RP_GT, ps_GT, propLoss=propLoss)

        BF_GT = Beamformer(RP=RP_GT)
        #BF_GT.Beamform(RP_GT, BI=range(0, RP_GT.numProj), soft=True, z=0.2)
        #BF_GT.display2Dscene(path='pics6/GT.png', show=False)

        RP_EST = RenderParameters(device=dev_1)
        #RP_EST.defineProjectorPosSpiral(show=False)
        # RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop, zStep=zStep)
        RP_EST.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_EST.generateTransmitSignal(compute=compute)
        print("Number of projectors")
        print(RP_EST.numProj)

        BF_EST = Beamformer(RP=RP_EST)

        vis = visdom.Visdom()
        #loss_window_big = vis.line(
        #    Y=torch.zeros((1)).cpu(),
        #    X=torch.zeros((1)).cpu(),
        #    opts=dict(xlabel='Iter', ylabel='Loss', title='Point 2 Point Test with Linear Window', legend=['']))
        loss_window_small = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iter', ylabel='Loss', title='Point 2 Point Test with Linear Window', legend=['']))
        wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))
        wass1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, diameter=1.0)

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    lr = .1
    cs = torch.nn.CosineSimilarity()
    #optimizer = torch.optim.SGD([ps_est], lr=lr)

    epochs = 100000
    Projs = range(0, RP_EST.numProj)
    batches = list(batchFromList(Projs, BS))
    losses = []

    wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
    noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

    for i in range(0, epochs):
        batch = random.sample(Projs, k=BS)

        #z = 0.0
        with torch.no_grad():
            pixels = np.random.choice(RP_EST.numPix3D, size=18000, replace=False)
            randPixels = RP_EST.pixPos3D[pixels, :]
            simulateWaveformsBatched(RP_GT, ps_GT, batch, propLoss=propLoss)
            GT = BF_GT.Beamform(RP_GT, BI=range(0, len(batch)), soft=True, pixels=randPixels).abs().to(dev_1)


        simulateWaveformsBatched(RP_EST, ps_est, batch, propLoss=propLoss)
        est = BF_EST.Beamform(RP_EST, BI=range(0, len(batch)), soft=True, pixels=randPixels).abs().to(dev_1)

        #BF_GT.display2Dscene(show=False, path='pics6/gt.png')
        #print("EST")
        #BF_EST.display2Dscene(show=True)

        est_pdf = est/torch.sum(est)
        GT_pdf = GT/torch.sum(GT)

        #pixels = torch.from_numpy(randPixels).to(dev_1)

        #loss = 1 - cs(est_pdf.unsqueeze(0), GT_pdf.unsqueeze(0))
        pixels = BF_EST.pixels.type(torch.float64)

        pixels = pixels.to(dev_1)

        loss = 100*wass1_loss(est_pdf.unsqueeze(1), pixels, GT_pdf.unsqueeze(1), pixels)
        #loss = torch.sum(torch.abs(est_pdf - GT_pdf))

        loss.backward()
        RP_GT.hooks.clear()
        RP_EST.hooks.clear()


        print("Position: " + str(ps_est.data.cpu().numpy()[0]) + '\t' + "Gradient: " + str(ps_est.grad.data.cpu().numpy()[0] * lr))

        ps_est.data -= noise.sample(ps_est.shape).squeeze().to(dev_1)
        ps_est.data -= lr*ps_est.grad

        dist = torch.norm(ps_est.detach().cpu() - ps_GT.detach().cpu(), p=2)
        grad_mag = torch.sqrt(torch.norm(ps_est.grad.data, p=2))

        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window_small, name = 'Wasserstein', update='append')
        vis.line(X=torch.ones((1)).cpu() * i, Y=grad_mag.unsqueeze(0).cpu(), win=loss_window_small, name = 'Grad Mag', update='append')
        vis.line(X=torch.ones((1)).cpu() * i, Y=dist.unsqueeze(0).cpu(), win=loss_window_small, name = 'Euc. Dist.', update='append')
        ps_est.grad.data.zero_()



