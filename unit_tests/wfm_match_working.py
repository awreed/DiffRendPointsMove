import torch
from RenderParameters import *
from simulateSASWaveformsPointSource import *
from VectorDistribution import *
from utils import *

RP_GT = RenderParameters()
RP_GT.generateTransmitSignal()
RP_GT.defineProjectorPos(thetaStart=0, thetaStop=0, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
ps_GT = torch.tensor([[-1.0, -1.0]], requires_grad=True).cuda()
simulateSASWaveformsPointSource(RP_GT, ps_GT)


RP_EST = RenderParameters()
RP_EST.generateTransmitSignal()
RP_EST.defineProjectorPos(thetaStart=0, thetaStop=0, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
ps_EST = torch.tensor([[-.5, -.5]], requires_grad=True).cuda()

print(RP_GT.Fs, RP_EST.Fs)


loss_val = 10000
thresh = 10
optimizer = torch.optim.SGD([ps_EST], lr=.0001, momentum=0)
#criterion = torch.nn.PairwiseDistance(p=1.0)
#k = torch.tensor([0.1])
loss = []
fig, axes = plt.subplots(1, 2)
while loss_val > thresh:
    simulateSASWaveformsPointSource(RP_EST, ps_EST)

    for pData_gt, pData_est in zip(RP_GT.projDataArray, RP_EST.projDataArray):
        #EX_GT = VectorDistribution(pData_gt.wfmRC.abs())
        #EX_EST = VectorDistribution(pData_est.wfmRC.abs())
        #auto_corr = xcorr(pData_gt.wfmRC.abs(), pData_gt.wfmRC.abs())
        #corr = xcorr(pData_gt.wfmRC.abs(), pData_est.wfmRC.abs())
        EX_GT = VectorDistribution(pData_gt.wfmRC.abs())
        EX_EST = VectorDistribution(pData_est.wfmRC.abs())
        #dist = torch.abs(auto_corr - corr))

        dist = torch.abs(EX_GT.mean - EX_EST.mean)
        #l1 = k * criterion(pData_gt.wfmRC.abs().unsqueeze(0), pData_est.wfmRC.abs().unsqueeze(0))
        loss.append(dist)
        plt.cla()
        axes[0].plot(pData_gt.wfmRC.abs().clone().detach().cpu().numpy(), color="blue")
        axes[0].plot(pData_est.wfmRC.abs().clone().detach().cpu().numpy(), color="red")

        plt.pause(0.05)
    axes[0].clear()
    axes[1].clear()
    final_loss = torch.sum(torch.stack(loss))
    #print(final_loss)
    final_loss.backward(retain_graph=True)
    #print(ps_EST.grad)

    optimizer.step()
    optimizer.zero_grad()
    loss.clear()

plt.show()