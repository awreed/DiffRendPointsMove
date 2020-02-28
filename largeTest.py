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

        GT = GT_Mesh.centroids[0, :]
        EST = EST_Mesh.centroids[0, :]
        GT = torch.from_numpy(GT).unsqueeze(0).cuda()
        EST = torch.from_numpy(EST).unsqueeze(0).cuda()

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
        #GT_BF = BF.beamformTest(RP_GT).abs()
        #GT_XY = GT_BF.view(x_vals, y_vals)
        #plt.imshow(GT_XY.detach().cpu().numpy())
        #plt.savefig("pics1/GT.png")

        # Estimate stuff
        RP_EST = RenderParameters()
        RP_EST.generateTransmitSignal()
        RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
        EST_Loc = torch.linspace(0, 1, int(RP_EST.nSamples)).view(-1, 1).cuda()
        EST_Loc_B = EST_Loc.repeat(BS, 1, 1)

        lr = .000001
        epochs = 100000

        wass_loss = SamplesLoss("sinkhorn", p=1, blur=.01)

        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='Wass-1 Loss', legend=['Loss']))

    ps_est = EST.clone()
    ps_est.requires_grad = True
    optimizer = torch.optim.SGD([ps_est], lr=lr, momentum=0.0)
    fig, axes = plt.subplots(1, 1)
    for i in range(0, epochs):
        #batch_indices = random.sample(range(0, RP_EST.numProj - 1), BS)
        batch_indices = [0]
        simulateWaveformsBatched(RP_EST, ps_est)

        loss = wass_loss(RP_GT.projDataArray[0].wfmRC.abs().unsqueeze(0), EST_Loc_B,
                         RP_EST.projDataArray[0].wfmRC.abs().unsqueeze(0), GT_Loc_B)

        final_loss = torch.sum(loss)
        final_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        axes.plot(RP_GT.projDataArray[0].wfmRC.abs().squeeze().clone().detach().cpu().numpy(), color='blue', alpha=0.5)
        axes.plot(RP_EST.projDataArray[0].wfmRC.abs().squeeze().clone().detach().cpu().numpy(), color='red', alpha=0.5)

        plt.pause(0.05)

        vis.line(X=torch.ones((1)).cpu() * i, Y=final_loss.unsqueeze(0).cpu(), win=loss_window, update='append')

        RP_EST.freeHooks()

        #if i % 1000 == 0:
        #    simulateWaveformsBatched(RP_EST, est)
        #    EST_BF = BF.beamformTest(RP_EST).abs().detach()
        #    EST_XY = EST_BF.view(x_vals, y_vals)
        #    plt.imshow(EST_XY.cpu().numpy())
        #    plt.savefig("pics1/est" + str(i) + ".png")
        axes.clear()
        axes.clear()
    plt.show()




