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
        torch.cuda.empty_cache()
    else:
        dev = "cpu"
    torch.cuda.empty_cache()

    device = torch.device(dev)
    BS = 1
    with torch.no_grad():
        GT_Mesh = ObjLoader('cube_bottom_64.obj')
        GT_Mesh.getCentroids()
        EST_Mesh = ObjLoader('cube_bottom_64.obj')
        EST_Mesh.vertices = EST_Mesh.vertices * 0.1
        EST_Mesh.getCentroids()

        GT = GT_Mesh.centroids
        EST = EST_Mesh.centroids
        GT = torch.from_numpy(GT).cuda()
        EST = torch.from_numpy(EST).cuda()

        # Ground truth stuff
        RP_GT = RenderParameters()
        RP_GT.generateTransmitSignal()
        RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=2, rStop=2, zStart=.3, zStop=.3)
        GT_Loc = torch.linspace(0, 1, int(RP_GT.nSamples)).view(-1, 1).cuda()
        GT_Loc_B = GT_Loc.repeat(BS, 1, 1)

        # Beamformer to create images from projector waveforms
        BF = Beamformer(sceneDimX=[-2, 2], sceneDimY=[-2, 2], sceneDimZ=[0, 0], nPix=[256, 256, 1], dim=2)
        x_vals = torch.unique(BF.pixPos[:, 0]).numel()
        y_vals = torch.unique(BF.pixPos[:, 1]).numel()
        simulateWaveformsBatched(RP_GT, GT)
        GT_BF = BF.beamformTest(RP_GT).abs()
        GT_XY = GT_BF.view(x_vals, y_vals)
        plt.imshow(GT_XY.detach().cpu().numpy())
        plt.savefig("pics1/GT.png")

        # Estimate stuff
        RP_EST = RenderParameters()
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
        EST_Loc = torch.linspace(0, 1, int(RP_EST.nSamples)).view(-1, 1).cuda()
        EST_Loc_B = EST_Loc.repeat(BS, 1, 1)

        lr = 1
        epochs = 100000

        wass_loss = SamplesLoss()

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='Wass-1 Loss', legend=['Loss']))

    est = EST.clone()
    est.requires_grad = True
    optimizer = torch.optim.SGD([est], lr=lr, momentum=0.0)
    fig, axes = plt.subplots(1, 2)
    for i in range(0, epochs):
        #batch_indices = random.sample(range(0, RP_EST.numProj - 1), BS)
        batch_indices = [0]
        simulateWaveformsBatched(RP_EST, est, batch_indices)
        EST_Wfms = []
        for pData_est in RP_EST.projDataArray:
            EST_Wfm = pData_est.wfmRC.abs() / torch.norm(pData_est.wfmRC.abs(), p=1)
            EST_Wfms.append(EST_Wfm)
        EST_Wfms_B = torch.stack(EST_Wfms)

        simulateWaveformsBatched(RP_GT, GT, batch_indices)
        # Normalize the GT waveforms
        GT_Wfms = []
        for pData_gt in RP_GT.projDataArray:
            GT_Wfm = pData_gt.wfmRC.abs() / torch.norm(pData_gt.wfmRC.abs(), p=1)
            GT_Wfms.append(GT_Wfm)
        GT_Wfms_B = torch.stack(GT_Wfms)

        #GT_Wfms_B = GT_Wfms_T[batch_indices, :]
        #EST_Wfms_B = EST_Wfms_T[batch_indices, :]

        loss = wass_loss(EST_Wfms_B, EST_Loc_B, GT_Wfms_B, GT_Loc_B)

        final_loss = torch.sum(loss)
        final_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        axes[0].plot(EST_Wfms_B.squeeze().clone().detach().cpu().numpy(), color="blue", alpha=0.5)
        axes[0].plot(GT_Wfms_B.squeeze().clone().detach().cpu().numpy(), color="red", alpha=0.5)

        plt.pause(0.05)

        vis.line(X=torch.ones((1)).cpu() * i, Y=final_loss.unsqueeze(0).cpu(), win=loss_window, update='append')

        if i % 1000 == 0:
            simulateWaveformsBatched(RP_EST, est)
            EST_BF = BF.beamformTest(RP_EST).abs().detach()
            EST_XY = EST_BF.view(x_vals, y_vals)
            plt.imshow(EST_XY.cpu().numpy())
            plt.savefig("pics1/est" + str(i) + ".png")
        axes[0].clear()
    plt.show()




