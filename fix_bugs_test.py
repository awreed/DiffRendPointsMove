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
from pytorch3d.loss import chamfer_distance
import time
import random
import collections
from batch_time_delay import *
import visdom
from Wavefront import *
import pickle
from UpsamplePixels import UpsamplePixels
import sys
from utils import *
np.set_printoptions(threshold=sys.maxsize)
import torch.nn.functional as F

#Matlab gradient_test.m under /home/albert

#Wrap up the 3D upsampling tests
# More investigation into the matlab code I was running
# Pretty sure this shit will work with an impossible projector geometry
# Would be really cool to show that as a result of tom


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

    BS = 360

    thetaStart = 0
    thetaStop = 359
    thetaStep = 1
    rStart = 1
    rStop = 1
    zStart = .5
    zStop = 1
    zStep = .01
    sceneDimX = [-.4, .4] # min dist needs to equal max dist
    sceneDimY = [-.4, .4]
    sceneDimZ = [-.4, .4]
    pixDim = [64, 64, 64]
    propLoss = False
    compute = False
    show=False

    with torch.no_grad():
        #GT_Mesh = ObjLoader('cube_64_bottom_z.obj')
        #GT_Mesh.vertices = GT_Mesh.vertices * 0.3
        #GT_Mesh.getCentroids()
        #EST_Mesh = ObjLoader('cube_64_bottom_z.obj')
        #EST_Mesh.vertices = EST_Mesh.vertices * 0.01
        #EST_Mesh.getCentroids()

        #GT = GT_Mesh.centroids

        #EST = EST_Mesh.centroids
        #ps_GT = torch.from_numpy(GT).to(dev)
        #ps_EST = torch.from_numpy(EST).to(dev_1)

        ps_GT = torch.tensor([[-.3, -.3, 0]], requires_grad=False).to(dev)
        ps_EST = torch.tensor([[.3, 0.3, 0]], requires_grad=False).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        #print("here")
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop, zStep=zStep)
        RP_GT.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_GT.generateTransmitSignal(compute=compute)
        print("Number of projectors")
        print(RP_GT.numProj)

        Projs = range(0, RP_GT.numProj)
        batch = random.sample(Projs, k=BS)
        BF_GT = Beamformer(RP=RP_GT)

        RP_EST = RenderParameters(device=dev_1)

        #RP_EST.defineProjectorPosSpiral(show=False)
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop, zStep=zStep)
        RP_EST.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_EST.generateTransmitSignal(compute=compute)
        print("Number of projectors")
        print(RP_EST.numProj)

        BF_EST = Beamformer(RP=RP_EST)

        vis = visdom.Visdom()

        loss_window_small = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iter', ylabel='Loss', title='Point 2 Point Test with Linear Window', legend=['']))
        wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))
        wass1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, diameter=1.0)

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    lr = .005
    w_l1 = 10
    w_cs = 100
    w_wass = 10
    cs = torch.nn.CosineSimilarity()
    optimizer = torch.optim.Adam([ps_est], lr=lr)

    epochs = 100000
    Projs = range(0, RP_EST.numProj)
    batches = list(batchFromList(Projs, BS))
    losses = []

    wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
    noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111, projection='3d')

    randPixels = RP_EST.pixPos3D


    for i in range(1, epochs):
        optimizer.zero_grad()
        batch = random.sample(Projs, k=BS)

        #z = 0.0
        with torch.no_grad():
            #pixels = np.random.choice(RP_EST.numPix3D, size=18000, replace=False)
            #randPixels = RP_EST.pixPos3D
            #randPixels = RP_EST.pixPos3D[pixels, :]
            simulateWaveformsBatched(RP_GT, ps_GT, batch, propLoss=propLoss)
            GT = BF_GT.Beamform(RP_GT, BI=range(0, len(batch)), soft=False, pixels=randPixels).abs().to(dev_1)
            #print(GT.dtype)

        simulateWaveformsBatched(RP_EST, ps_est, batch, propLoss=propLoss)
        est = BF_EST.Beamform(RP_EST, BI=range(0, len(batch)), soft=False, pixels=randPixels).abs().to(dev_1)

        #BF_EST.displayScene(ax=ax)

        est_norm = est/torch.sum(est)
        GT_norm = GT/torch.sum(GT)

        est_scaled = (est_norm - est_norm.min())/(est_norm.max() - est_norm.min())
        GT_scaled = (GT_norm - GT_norm.min())/(GT_norm.max() - GT_norm.min())

        #k = 50
        k = 4

        est_scaled[est_scaled[:] < (torch.mean(est_scaled) + k * torch.std(est_scaled))] = 0
        GT_scaled[GT_scaled[:] < (torch.mean(GT_scaled) + k * torch.std(GT_scaled))] = 0

        est_scaled.retain_grad()

        #est_scaled.retain_grad()

        #est_scaled.retain_grad()

        #l1_loss = w_l1*torch.sum(torch.sqrt((est_norm - GT_norm)**2))
        #cs_loss = w_cs * (1 - cs(GT_norm.unsqueeze(0), est_norm.unsqueeze(0)))
        #pixels = torch.from_numpy(randPixels).to(dev_1)

        pixels = BF_EST.pixels.type(torch.float64)
        pixels = pixels.to(dev_1)

        x_vals = torch.unique(pixels[:, 0]).numel()
        y_vals = torch.unique(pixels[:, 1]).numel()

        #wass_loss = w_wass*wass1_loss(GT_scaled.unsqueeze(1), pixels, est_scaled.unsqueeze(1), pixels)

        #print("EST")
        #print(est_scaled.detach().cpu().numpy())
        #print("GT")
        #print(GT_scaled.detach().cpu().numpy())

        l1_loss = w_l1 * torch.sum(((est_scaled - GT_scaled)**2))

        loss = l1_loss

        loss.backward()

        #grad = (e.grad - est_scaled.grad.min())/est_scaled.grad.max()
        #print("Est scaled gradient")
        #x = est_scaled.grad.detach().cpu().numpy()
        #print(x)

        #BF_EST.sideByside(img1=GT_scaled, img2=est_scaled, img3=est_scaled.grad, path='pics6/GT_est' + str(i) + '.png', show=False)

        torch.cuda.set_device(dev)
        loss_chamfer, _ = chamfer_distance(ps_GT.unsqueeze(0), ps_est.unsqueeze(0).to(dev))
        loss_chamfer = loss_chamfer * 10
        RP_GT.freeHooks()
        RP_EST.freeHooks()

        optimizer.step()
        print(ps_est.data)
        ps_numpy = ps_est.data.detach().cpu().numpy()

        #if i % 300 == 0:
            #text = input("Upsample pixels?")
            #if text == 'y':
            #with torch.no_grad():
                #simulateWaveformsBatched(RP_EST, ps_est, range(0, RP_EST.numProj), propLoss=propLoss)
                #randPixels = UpsamplePixels(RP=RP_EST, BF=BF_EST, batch=batch)
                #print("YES")
            #elif text == 'n':
            #    print('NO')
            #else:
            #    print("What")

        #vis.line(X=torch.ones((1)).cpu() * i, Y=wass_loss.unsqueeze(0).cpu(), win=loss_window_small, name = 'Wasserstein', update='append')
        #vis.line(X=torch.ones((1)).cpu() * i, Y=wass_loss.unsqueeze(0).cpu(), win=loss_window_small, name = 'Wass Loss', update='append')
        #vis.line(X=torch.ones((1)).cpu() * i, Y=loss_chamfer.unsqueeze(0).cpu(), win=loss_window_small, name = 'CS Loss', update='append')
        #vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window_small, name='Total Loss',
                 #update='append')

        ax1.clear()
        ax1.scatter(ps_numpy[:, 0], ps_numpy[:, 1], ps_numpy[:, 2])
        ax1.set_xlim3d((-.4, .4))
        ax1.set_ylim3d((-.4, .4))
        ax1.set_zlim3d((-.4, .4))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.pause(.05)


        #ps_est.grad.data.zero_()



