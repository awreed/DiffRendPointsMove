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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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


# When emailing Tom:
# Object size is important
# Smaller scene saves memory if I range gate

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
    thetaStart = 0
    thetaStop = 359
    thetaStep = 1
    rStart = 1
    rStop = 1
    zStart = -0.4
    zStop = 0.4
    zStep = 0.001
    sceneDimX = [-.4, .4] # min dist needs to equal max dist
    sceneDimY = [-.4, .4]
    sceneDimZ = [-.4, .4]
    pixDim = [128, 128, 128]
    propLoss = False
    compute = False

    with torch.no_grad():
        ps_GT = torch.tensor([[-.3, -.3, -0.3]], requires_grad=False).to(dev)
        ps_EST = torch.tensor([[.3, .3, -0.3]], requires_grad=False).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep,
                                 rStart=rStart, rStop=rStop, zStart=zStart, zStop=zStop, zStep=zStep)
        RP_GT.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_GT.generateTransmitSignal(compute=compute)
        print("Number of projectors")
        print(RP_GT.numProj)

        #simulateWaveformsBatched(RP_GT, ps_GT, propLoss=propLoss)

        BF_GT = Beamformer(RP=RP_GT)
        #BF_GT.Beamform(RP_GT, BI=range(0, RP_GT.numProj), soft=True, z=0.2)
        #BF_GT.display2Dscene(path='pics6/GT.png', show=False)

        RP_EST = RenderParameters(device=dev_1)
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


        lr = .1
        cs = torch.nn.CosineSimilarity()
        #optimizer = torch.optim.SGD([ps_est], lr=lr)

        epochs = 100000
        Projs = range(0, RP_EST.numProj)
        batches = list(batchFromList(Projs, BS))
        losses = []

        xs = np.linspace(-.3, .3, 10)
        ys = np.linspace(-.3, .3, 10)
        zs = np.linspace(-.3, .3, 10)

        (x, y, z) = np.meshgrid(xs, ys, zs)

        points = np.hstack((np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)),
                                   np.reshape(z, (np.size(z), 1))))

        points = torch.from_numpy(points)
        points.requires_grad = False

        losses = []
        count = 1

        for ps_est in points:
            ps_est = ps_est.unsqueeze(0).to(dev_1)
            batch = random.sample(Projs, k=BS)

            simulateWaveformsBatched(RP_EST, ps_est, batch, propLoss=propLoss)
            simulateWaveformsBatched(RP_GT, ps_GT, batch, propLoss=propLoss)

            #z = np.random.choice(RP_EST.zVect)
            pixels = np.random.choice(RP_EST.numPix3D, size=18000, replace=False)
            randPixels = RP_EST.pixPos3D[pixels, :]

            est = BF_EST.Beamform(RP_EST, BI=range(0, len(batch)), soft=True, pixels=randPixels).abs().to(dev_1)
            GT = BF_GT.Beamform(RP_GT, BI=range(0, len(batch)), soft=True, pixels=randPixels).abs().to(dev_1)

            #BF_EST.display2Dscene(path='pics6/est' + str(i) + '.png', show=False)

            #est_pdf = est/torch.sum(est)
            #GT_pdf = GT/torch.sum(GT)
            est_pdf = est/torch.norm(est, p=2)
            GT_pdf = GT/torch.norm(GT, p=2)

            pixels = torch.from_numpy(randPixels).to(dev_1)

            losses.append(torch.sum(torch.abs(est_pdf - GT_pdf)))
            print(count)
            count += 1

        val = torch.stack(losses).squeeze()
        print(val.shape)
        val = val.detach().cpu().numpy()
        print(val.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(x, y, z, c=val, alpha=0.5, cmap=plt.hot())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(img)
        plt.show()










