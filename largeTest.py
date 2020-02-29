from geomloss import SamplesLoss
from batch_time_delay import *
from RenderParameters import *
from Wavefront import *
import numpy as np
import random
import visdom
from Beamformer import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
        dev1 = "cuda:1"
    else:
        dev = "cpu"

    device = torch.device(dev)
    BS = 10
    with torch.no_grad():
        GT_Mesh = ObjLoader('cube_bottom_64.obj')
        GT_Mesh.getCentroids()
        EST_Mesh = ObjLoader('cube_bottom_64.obj')
        EST_Mesh.vertices = EST_Mesh.vertices * 0.1
        EST_Mesh.getCentroids()

        GT = GT_Mesh.centroids
        #GT = torch.tensor([[0, .5], [0, -.5]], requires_grad=False).cuda()
        #EST = torch.tensor([[-.5, -.5], [-.25, -.25]], requires_grad=False).cuda()
        EST = EST_Mesh.centroids
        GT = torch.from_numpy(GT).cuda()
        EST = torch.from_numpy(EST).cuda()

        # Ground truth stuff
        RP_GT = RenderParameters()
        RP_GT.generateTransmitSignal()
        RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
        GT_Loc = torch.linspace(0, 1, int(RP_GT.nSamples)).view(-1, 1).cuda()
        GT_Loc_B = GT_Loc.repeat(BS, 1, 1).to(dev1)
        simulateWaveformsBatched(RP_GT, GT)
        GT_Wfms = []
        for pData_gt in RP_GT.projDataArray:
            GT_Wfms.append(pData_gt.normWfmRC)
        GT_Wfms_T = torch.stack(GT_Wfms)


        # Beamformer to create images from projector waveforms
        BF = Beamformer(sceneDimX=[-2, 2], sceneDimY=[-2, 2], sceneDimZ=[0, 0], nPix=[256, 256, 1], dim=2)
        x_vals = torch.unique(BF.pixPos[:, 0]).numel()
        y_vals = torch.unique(BF.pixPos[:, 1]).numel()

        GT_BF = BF.beamformTest(RP_GT).abs()
        GT_XY = GT_BF.view(x_vals, y_vals)
        plt.imshow(GT_XY.detach().cpu().numpy())
        plt.savefig("pics1/GT.png")
        del RP_GT
        del GT_BF
        del GT_XY

        # Estimate stuff
        RP_EST = RenderParameters()
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
        EST_Loc = torch.linspace(0, 1, int(RP_EST.nSamples)).view(-1, 1).cuda()
        EST_Loc_B = EST_Loc.repeat(BS, 1, 1).to(dev1)

        lr = .1
        epochs = 100000

        wass_loss = SamplesLoss("sinkhorn", p=1, blur=.01)
        wiggle = 2.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='Wass-1 Loss', legend=['Loss']))

    ps_est = EST.clone()
    ps_est.requires_grad = True
    fig, axes = plt.subplots(1, 1)
    for i in range(0, epochs):
        batch_indices = random.sample(range(0, RP_EST.numProj - 1), BS)
        simulateWaveformsBatched(RP_EST, ps_est, batch_indices)

        EST_Wfms = []
        for pData_est in RP_EST.projDataArray:
            EST_Wfms.append(pData_est.normWfmRC)
        EST_Wfms_B = torch.stack(EST_Wfms).to(dev1)

        GT_Wfms_B = GT_Wfms_T[batch_indices, :].to(dev1)

        loss = wass_loss(GT_Wfms_B, EST_Loc_B,
                         EST_Wfms_B, GT_Loc_B)

        final_loss = torch.sum(loss)
        print(final_loss)
        final_loss.backward()

        ps_est.data += noise.sample(ps_est.shape).squeeze()
        ps_est.data -= lr * ps_est.grad
        ps_est.grad.zero_()

        vis.line(X=torch.ones((1)).cpu() * i, Y=final_loss.unsqueeze(0).cpu(), win=loss_window, update='append')
        del loss
        del final_loss
        del EST_Wfms_B
        del GT_Wfms_B

        if i % 100 == 0:
            torch.cuda.empty_cache()
            simulateWaveformsBatched(RP_EST, ps_est)
            EST_BF = BF.beamformTest(RP_EST).abs().detach()
            EST_XY = EST_BF.view(x_vals, y_vals)
            plt.imshow(EST_XY.cpu().numpy())
            plt.savefig("pics1/est" + str(i) + ".png")
            del EST_BF
            del EST_XY
        #axes.clear()
        #axes.clear()
    #plt.show()




