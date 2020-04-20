import torch
from RenderParameters import *
from simulateSASWaveformsPointSource import *
from VectorDistribution import *
from utils import *
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from batch_time_delay import *

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

with torch.no_grad():
    RP_GT = RenderParameters()
    RP_GT.generateTransmitSignal()
    RP_GT.defineProjectorPos(thetaStart=0, thetaStop=0, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    ps_GT = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], requires_grad=False).cuda()
    simulateWaveformsBatched(RP_GT, ps_GT)
    GT_Wfm1 = RP_GT.projDataArray[0].wfmRC.abs()

    RP_EST = RenderParameters()
    RP_EST.generateTransmitSignal()
    RP_EST.defineProjectorPos(thetaStart=0, thetaStop=0, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    ps_EST = torch.tensor([[0, .5], [0, .5]], requires_grad=False).cuda()

    GT_Loc = torch.linspace(0, 1, len(GT_Wfm1)).view(-1, 1).cuda()
    EST_Loc = torch.linspace(0, 1, len(GT_Wfm1)).view(-1, 1).cuda()

ps = ps_EST.clone()
ps.requires_grad = True


loss_val = 10000
thresh = 10
lr = .1
#optimizer = torch.optim.SGD([ps], lr=lr, momentum=0)
wass_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, diameter=1.0)
wiggle = 2.0*(1/RP_GT.Fs)*RP_GT.c
noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

losses = []
fig, axes = plt.subplots(1, 1)
epochs = 500
for i in range(0, epochs):
    simulateWaveformsBatched(RP_EST, ps)

    for pData_gt, pData_est in zip(RP_GT.projDataArray, RP_EST.projDataArray):
        GT_Wfm1 = pData_gt.wfmRC.abs()
        GT_WfmNorm = GT_Wfm1 / torch.norm(GT_Wfm1, p=1)
        GT_Wfm = GT_WfmNorm

        EST_Wfm1 = pData_est.wfmRC.abs()
        EST_WfmNorm = EST_Wfm1/torch.norm(EST_Wfm1, p=1)
        EST_Wfm = EST_WfmNorm

        #est_wfm = VectorDistribution(EST_Wfm1)
        #gt_wfm = VectorDistribution(GT_Wfm1)
        #loss = torch.abs(est_wfm.mean - gt_wfm.mean)

        print(EST_Wfm.shape)
        print(EST_Loc.shape)

        loss = wass_loss(EST_Wfm, EST_Loc, GT_Wfm, GT_Loc)
        #print(p1.requires_grad)
        #print(p2.requires_grad)
        #loss = torch.sum(p1 + p2)
        losses.append(loss)

        axes.plot(EST_Wfm.clone().detach().cpu().numpy(), color="blue")
        axes.plot(GT_Wfm.clone().detach().cpu().numpy(), color="red")

        plt.pause(0.05)
        plt.savefig("../figs/fig" + str(i) + ".png", dpi=fig.dpi)

    axes.clear()
    final_loss = torch.sum(torch.stack(losses))
    print(final_loss)

    final_loss.backward()
    #ps.data += noise.sample(ps.shape).squeeze()
    ps.data -= lr*ps.grad
    ps.grad.zero_()
    losses.clear()

plt.show()





