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
    BS = 40
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
    with torch.no_grad():
        GT_Mesh = ObjLoader('cube_bottom_64.obj')
        GT_Mesh.vertices = GT_Mesh.vertices * 0.3
        GT_Mesh.getCentroids()
        EST_Mesh = ObjLoader('cube_bottom_64.obj')
        EST_Mesh.vertices = EST_Mesh.vertices * 0.01
        EST_Mesh.getCentroids()

        GT = GT_Mesh.centroids

        EST = EST_Mesh.centroids
        ps_GT = torch.from_numpy(GT).to(dev)
        ps_EST = torch.from_numpy(EST).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        RP_GT.generateTransmitSignal()
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                 rStop=rStop, zStart=zStart, zStop=zStop)
        #ps_GT = torch.tensor([[-.3, -.3], [.3, -.3]], requires_grad=False).to(dev)
        simulateWaveformsBatched(RP_GT, ps_GT, propLoss=True)
        #simulateSASWaveformsPointSource(RP_GT, ps_GT, gt=False)
        print(RP_GT.numProj)

        # Beamformer to create images from projector wavef orms
        BF_GT = Beamformer(RP=RP_GT, sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, nPix=nPix, dim=2)
        x_vals = torch.unique(BF_GT.pixPos[:, 0]).numel()
        y_vals = torch.unique(BF_GT.pixPos[:, 1]).numel()
        #GT = BF_GT.softBeamformer(RP_GT).abs().to(dev_1)
        GT_hard = BF_GT.softBeamformer(RP_GT, BI=range(0, RP_GT.numProj), soft=False).abs().to(dev_1)
        GT_XY = GT_hard.view(x_vals, y_vals)
        #plt.imshow(GT_XY.detach().cpu().numpy())
        plt.stem(RP_GT.projDataArray[0].wfmRC.abs().detach().cpu().numpy(), use_line_collection=True)
        plt.savefig("pics1/GT.png")

        RP_EST = RenderParameters(device=dev_1)
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop)

        BF_EST = Beamformer(RP=RP_EST, sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, nPix=nPix, dim=2)

        #ps_EST = torch.tensor([[.3, .3], [-.3, .3]], requires_grad=False).to(dev_1)
        #ps_EST = torch.tensor([[-.2, -.2], [.2, -.2]], requires_grad=False).to(dev_1)

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='Wass-1 Loss', legend=['Loss']))
        wiggle = 2.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    optimizer = torch.optim.Adam([ps_est], lr=0.01)

    epochs = 1000

    for i in range(0, epochs):
        batch = random.sample(range(0, RP_EST.numProj - 1), BS)
        simulateWaveformsBatched(RP_EST, ps_est, batch, propLoss=True)
        #simulateSASWaveformsPointSource(RP_EST, ps_est)

        est = BF_EST.softBeamformer(RP_EST, BI=range(0, BS), soft=True).abs()
        GT = BF_GT.softBeamformer(RP_GT, BI=batch, soft=True).abs().to(dev_1)

        #est_xy = est.view(x_vals, y_vals)
        #plt.imshow(est_xy.detach().cpu().numpy())
        #plt.savefig("pics1/est" + str(i) + ".png")
        #plt.pause(.05)

        GT_norm = GT/torch.norm(GT, p=1)
        est_norm = est/torch.norm(est, p=1)

        loss = torch.sum(torch.sqrt((GT_norm - est_norm)**2))
        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #ps_est.data += noise.sample(ps_est.shape).squeeze().to(dev_1)

        if i % 10 == 0:
            with torch.no_grad():
                simulateWaveformsBatched(RP_EST, ps_est, propLoss=True)
                est_hard = BF_EST.softBeamformer(RP_EST, BI=range(0, RP_EST.numProj), soft=False).abs()

                #plt.show()
                #plt.clf()
                #plt.stem(RP_EST.projDataArray[0].wfmRC.abs().detach().cpu().numpy(), use_line_collection=True)
                #plt.show()
                est_xy = est_hard.view(x_vals, y_vals)
                plt.imshow(est_xy.detach().cpu().numpy())
                plt.savefig("pics1/est" + str(i) + ".png")

        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window, update='append')
