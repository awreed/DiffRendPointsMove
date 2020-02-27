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
        torch.cuda.empty_cache()
    else:
        print("Fix ur cuda loser")
        dev = "cpu"

    device = torch.device(dev)
    BS = 100

    # Data structure for ground truth projector waveforms
    RP_GT = RenderParameters()
    RP_GT.generateTransmitSignal()
    RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=2, rStop=2, zStart=.3, zStop=.3)
    ps_GT = torch.tensor([[-1.01, -2.01], [1.02, 1.02], [-2.0, 0.5], [2.03, -1.03]], requires_grad=True).cuda()
    GT_Loc = torch.linspace(0, (RP_GT.nSamples-1), int(RP_GT.nSamples)).view(-1, 1).cuda()
    GT_Loc_B = GT_Loc.repeat(BS, 1, 1)
    simulateWaveformsBatched(RP_GT, ps_GT)
    GT_Wfms = []
    for pData_gt in RP_GT.projDataArray:
        GT_Wfm = pData_gt.wfmRC.abs() / torch.norm(pData_gt.wfmRC.abs(), p=1)
        GT_Wfms.append(GT_Wfm)
    GT_Wfms_T = torch.stack(GT_Wfms)

    # Data structure for estimate projector waveforms
    RP_EST = RenderParameters()
    RP_EST.generateTransmitSignal()
    RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    ps_EST = torch.tensor([[0.11, 0.11], [-0.12, -0.12], [-0.1, 0.1], [0.13, -0.13]], requires_grad=True).cuda()
    EST_Loc = torch.linspace(0, (RP_EST.nSamples-1), int(RP_EST.nSamples)).view(-1, 1).cuda()
    EST_Loc_B = EST_Loc.repeat(BS, 1, 1)

    # Beamformer to create images from projector waveforms
    # BF = Beamformer(sceneDimX=[-2, 2], sceneDimY=[-2, 2], sceneDimZ=[0, 0], nPix=[128, 128, 1], dim=2)
    # x_vals = torch.unique(BF.pixPos[:, 0]).numel()
    # y_vals = torch.unique(BF.pixPos[:, 1]).numel()
    # GT = BF.beamformTest(RP_GT).abs()
    # GT_XY = GT.view(x_vals, y_vals)
    # plt.imshow(GT_XY.detach().cpu().numpy())
    # plt.savefig("pics1/GT.png")

    loss_val = 10000
    thresh = 10
    lr = .00005
    optimizer = torch.optim.SGD([ps_EST], lr=lr, momentum=0.000)
    wass_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, scaling=.05)
    epochs = 500
    lr_sched = 20
    losses = collections.deque(maxlen=lr_sched)

    vis = visdom.Visdom()
    loss_window = vis.line(
        Y=torch.zeros((1)).cpu(),
        X=torch.zeros((1)).cpu(),
        opts=dict(xlabel='epoch', ylabel='Loss', title='Wass-1 Loss', legend=['Loss']))

    fig, axes = plt.subplots(1, 1)
    for i in range(0, epochs):
        #batch_indices = random.sample(range(0, RP_EST.numProj - 1), BS)
        batch_indices = [0]
        simulateWaveformsBatched(RP_EST, ps_EST, batch_indices)
        EST_Wfms = []
        for pData_est in RP_EST.projDataArray:
            EST_Wfm = pData_est.wfmRC.abs() / torch.norm(pData_est.wfmRC.abs(), p=1)
            EST_Wfms.append(EST_Wfm)
        EST_Wfms_B = torch.stack(EST_Wfms)

        # EST_Wfms_B = EST_Wfms_T[batch_indices, :]
        GT_Wfms_B = GT_Wfms_T[batch_indices, :]

        loss = wass_loss(EST_Wfms_B, EST_Loc_B, GT_Wfms_B, GT_Loc_B)

        final_loss = torch.sum(loss)
        final_loss.backward(retain_graph=True)
        losses.append(final_loss.item())

        axes.plot(GT_Wfms_B.squeeze(0).detach().cpu().numpy(), color='red')
        axes.plot(EST_Wfms_B.squeeze(0).detach().cpu().numpy(), color='blue')
        plt.pause(0.05)
        axes.clear()

        # if len(losses) == lr_sched:
        #    if losses[-1] > losses[0]:
        #        for param_group in optimizer.param_groups:
        #            lr = lr * 0.75
        #            param_group['lr'] = lr
        #            losses.clear()
        print(ps_EST)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        vis.line(X=torch.ones((1)).cpu()*i, Y=final_loss.unsqueeze(0).cpu(), win=loss_window, update='append')
    plt.show()
        # Beamform estimate image
        # EST = BF.beamformTest(RP_EST).abs()
        # EST_XY = EST.view(x_vals, y_vals)

        # plt.imshow(EST_XY.detach().cpu().numpy())
        # plt.savefig("pics1/est" + str(i) + ".png")
