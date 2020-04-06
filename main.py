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

#When emailing Tom:
#Object size is important
#Smaller scene saves memory if I range gate

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
    propLoss=True
    with torch.no_grad():
        #GT_Mesh = ObjLoader('bunny.obj')
        #GT_Mesh.vertices[:, 2] = GT_Mesh.vertices[:, 2] + .06
        #GT_Mesh.vertices = GT_Mesh.vertices * 1.5
        #GT_Mesh.getCentroids()
        #EST_Mesh = ObjLoader('bunny.obj')
        #EST_Mesh.vertices = EST_Mesh.vertices * 0.01
        #EST_Mesh.getCentroids()

        #GT = GT_Mesh.centroids

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(GT[:, 0], GT[:, 1], GT[:, 2])
        #ax.set_xlim(sceneDimX[0], sceneDimX[1])
        ##ax.set_ylim(sceneDimY[0], sceneDimY[1])
        #ax.set_zlim(sceneDimZ[0], sceneDimZ[1])
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')

        #plt.savefig("pics6/GT_pc.png")

        #EST = EST_Mesh.centroids
        #ps_GT = torch.from_numpy(GT).to(dev)
        #ps_EST = torch.from_numpy(EST).to(dev_1)
        #ps_GT = torch.tensor([[-.3, -.3, 0.0], [0.3, -.3, 0.0]], requires_grad=False).to(dev)
        #ps_EST = torch.tensor([[-.2, -.2, 0.0], [.2, -.2, 0.0]], requires_grad=False).to(dev_1)
        ps_GT = torch.tensor([[-.3, -.3, 0.0]], requires_grad=False).to(dev)
        ps_EST = torch.tensor([[.3, .3, 0.0]], requires_grad=False).to(dev_1)

        # Data structure for ground truth projector waveforms
        RP_GT = RenderParameters(device=dev)
        RP_GT.generateTransmitSignal()
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                 rStop=rStop, zStart=zStart, zStop=zStop)
        simulateWaveformsBatched(RP_GT, ps_GT, propLoss=propLoss)
        print(RP_GT.numProj)

        # Beamformer to create images from projector waveforms
        BF_GT = Beamformer(RP=RP_GT, sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, nPix=nPix)
        x_vals = torch.unique(BF_GT.pixPos[:, 0]).numel()
        y_vals = torch.unique(BF_GT.pixPos[:, 1]).numel()
        GT_hard = BF_GT.Beamformer(RP_GT, BI=range(0, RP_GT.numProj), soft=True).abs().to(dev_1)
        #BF_GT.displayScene(dim=3, mag=True)

        GT_XY = GT_hard.view(x_vals, y_vals)
        plt.clf()
        plt.imshow(GT_XY.detach().cpu().numpy())
        plt.savefig("pics6/GT.png")
        #plt.clf()
        #plt.stem(data, use_line_collection=True)
        #plt.savefig("pics6/GTWfm.png")

        RP_EST = RenderParameters(device=dev_1)
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop)

        BF_EST = Beamformer(RP=RP_EST, sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, nPix=nPix)

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='L1 Loss', legend=['Loss']))
        wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

    ps_est = ps_EST.clone()
    ps_est.requires_grad = True
    optimizer = torch.optim.Adam([ps_est], lr=.01)

    epochs = 100000
    Projs = range(0, RP_EST.numProj)
    batches = list(batchFromList(Projs, BS))
    losses = []

    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    for i in range(0, epochs):
        #for batch in batches:
        batch = random.sample(Projs, k=BS)
        simulateWaveformsBatched(RP_EST, ps_est, batch, propLoss=propLoss)

        est = BF_EST.Beamformer(RP_EST, BI=range(0, len(batch)), soft=True).abs().to(dev_1)
        GT = BF_GT.Beamformer(RP_GT, BI=batch, soft=True).abs().to(dev_1)

        est_norm = est/torch.norm(est, p=1)
        GT_norm = GT/torch.norm(GT, p=1)



        #est_norm = (est - torch.mean(est))/torch.std(est)
        #GT_norm = (GT - torch.mean(GT))/torch.std(GT)

        #est_norm = est_pdf/est_pdf.max()
        #GT_norm = GT_pdf/GT_pdf.max()

        #axs[0].clear()
        #axs[1].clear()

        #axs[0].hist(est_norm.detach().cpu().numpy(), bins=100)
        #axs[1].hist(GT_norm.detach().cpu().numpy(), bins=100)

        #plt.pause(0.5)

        loss = torch.sum(torch.abs(est_norm - GT_norm))
            #losses.append(loss)
        loss.backward()

        #final_loss = torch.sum(torch.stack(losses))
        print(ps_est)
        #losses.clear()
        #print(loss.data)
        optimizer.step()
        optimizer.zero_grad()
        #ps_est.data += noise.sample(ps_est.shape).squeeze().to(dev_1)

        #points = ps_est.detach().cpu().numpy()
        #ax.clear()
        #ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        #ax.set_xlim(sceneDimX[0], sceneDimX[1])
        #ax.set_ylim(sceneDimY[0], sceneDimY[1])
        #ax.set_zlim(sceneDimZ[0], sceneDimZ[1])
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #plt.pause(.01)

        if i % 10 == 0:
            with torch.no_grad():
                #fig = plt.figure()
                #ax = fig.add_subplot(111, projection='3d')
                #points = ps_est.detach().cpu().numpy()
                #ax.scatter(points[:, 0], points[:, 1], points[:, 2])
                #ax.set_xlim(sceneDimX[0], sceneDimX[1])
                #ax.set_ylim(sceneDimY[0], sceneDimY[1])
                #ax.set_zlim(sceneDimZ[0], sceneDimZ[1])
                #ax.set_xlabel('X')
                #ax.set_ylabel('Y')
                #ax.set_zlabel('Z')
                #plt.show()

                simulateWaveformsBatched(RP_EST, ps_est, propLoss=propLoss)
                est = BF_EST.Beamformer(RP_EST, BI=range(0, RP_EST.numProj), soft=True).abs()


                #est_norm = (est - torch.mean(est))/torch.std(est)

                plt.clf()
                est_xy = est_norm.view(x_vals, y_vals)
                plt.imshow(est_xy.detach().cpu().numpy())
                plt.savefig("pics6/est" + str(i) + ".png")
                #plt.clf()
                #data = RP_EST.projDataArray[0].wfmRC.abs().detach().cpu().numpy()
                #plt.stem(data, use_line_collection=True)
                #plt.ylim((0, 1))
                #plt.show()
                #plt.savefig("pics6/estWfm" + str(i) + ".png")


                #ps = ps_est.detach().cpu().numpy()
                #fig = plt.figure()
                #ax = fig.add_subplot(111, projection='3d')
                #ax.set_xlabel('X')
                #ax.set_ylabel('Y')
                #ax.set_zlabel('Z')
                #ax.set_xlim(-.6, .6)
                #ax.set_ylim(-.6, .6
                #ax.set_zlim(-.6, .6)

                #ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2])
                #pickle.dump(fig, open('pics6/FigureObject' + str(i) + '.fig.pickle', 'wb'))
                #plt.savefig("pics6/est_pc" + str(i) + ".png")
                #plt.close(fig)

                #del fig


        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window, update='append')
