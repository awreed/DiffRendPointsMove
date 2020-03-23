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

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
    BS = 30
    thetaStart=0
    thetaStop=359
    thetaStep=1
    rStart=1
    rStop=1
    zStart=0.3
    zStop=0.3
    sceneDimX = [-.4, .4]
    sceneDimY = [-.4, .4]
    sceneDimZ = [0, 0]
    nPix = [128, 128, 1]
    dim=3
    with torch.no_grad():
        GT_Mesh = ObjLoader('cube768.obj')
        GT_Mesh.vertices = GT_Mesh.vertices * 0.3
        GT_Mesh.getCentroids()
        EST_Mesh = ObjLoader('cube768.obj')
        EST_Mesh.vertices = EST_Mesh.vertices * 0.01
        EST_Mesh.getCentroids()

        GT = GT_Mesh.centroids
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(GT[:, 0], GT[:, 1], GT[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-.6, .6)
        ax.set_ylim(-.6, .6)
        ax.set_zlim(-.6, .6)

        plt.savefig("pics2/GT_pc.png")

        EST = EST_Mesh.centroids
        ps_GT = torch.from_numpy(GT).to(dev)
        ps_EST = torch.from_numpy(EST).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        RP_GT.generateTransmitSignal()
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                 rStop=rStop, zStart=zStart, zStop=zStop)
        simulateWaveformsBatched(RP_GT, ps_GT, propLoss=True)
        print(RP_GT.numProj)

        # Beamformer to create images from projector wavef orms
        BF_GT = Beamformer(RP=RP_GT, sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, nPix=nPix, dim=dim)
        x_vals = torch.unique(BF_GT.pixPos[:, 0]).numel()
        y_vals = torch.unique(BF_GT.pixPos[:, 1]).numel()
        GT_hard = BF_GT.Beamformer(RP_GT, BI=range(0, RP_GT.numProj), soft=False).abs().to(dev_1)
        GT_XY = GT_hard.view(x_vals, y_vals)
        plt.clf()
        plt.imshow(GT_XY.detach().cpu().numpy())
        plt.savefig("pics2/GT.png")

        RP_EST = RenderParameters(device=dev_1)
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop)

        BF_EST = Beamformer(RP=RP_EST, sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, nPix=nPix, dim=dim)

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='L1 Loss', legend=['Loss']))
        wiggle = 2.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    optimizer = torch.optim.Adam([ps_est], lr=0.01)

    epochs = 10000


    for i in range(0, epochs):
        batch = random.sample(range(0, RP_EST.numProj - 1), BS)
        simulateWaveformsBatched(RP_EST, ps_est, batch, propLoss=True)

        est = BF_EST.Beamformer(RP_EST, BI=range(0, BS), soft=True)
        GT = BF_GT.Beamformer(RP_GT, BI=batch, soft=True)

        GT_mag = (GT.abs()/torch.norm(GT.abs(), p=1)).to(dev_1)

        est_mag = (est.abs()/torch.norm(est.abs(), p=1)).to(dev_1)

        loss = torch.sum(torch.abs(est_mag - GT_mag))
        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ps_est.data += noise.sample(ps_est.shape).squeeze().to(dev_1)

        if i % 10 == 0:
            with torch.no_grad():
                simulateWaveformsBatched(RP_EST, ps_est, propLoss=True)
                est_hard = BF_EST.Beamformer(RP_EST, BI=range(0, RP_EST.numProj), soft=False).abs()
                plt.clf()
                est_xy = est_hard.view(x_vals, y_vals)
                plt.imshow(est_xy.detach().cpu().numpy())
                plt.savefig("pics2/est" + str(i) + ".png")
                ps = ps_est.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-.6, .6)
                ax.set_ylim(-.6, .6)
                ax.set_zlim(-.6, .6)

                ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2])
                pickle.dump(fig, open('FigureObject' + str(i) + '.fig.pickle', 'wb'))
                plt.savefig("pics2/est_pc" + str(i) + ".png")
                plt.close(fig)

                del fig


        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window, update='append')
