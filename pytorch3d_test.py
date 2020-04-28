import os
import torch

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj, save_obj
from RenderParameters import *
from Beamformer import *
from batch_time_delay import *
from utils import *
import visdom
import random
from geomloss import SamplesLoss

np.random.seed(0)
random.seed(0)

def plot_pointcloud(mesh, fig, ax, angle, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    ax.clear()
    ax.scatter3D(x, z, -y)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim(-.4, .4)
    ax.set_ylim(-.4, .4)
    ax.set_zlim(-.4, .4)
    ax.set_title(title)
    ax.view_init(190, angle)
    plt.pause(.05)

def getMesh(**kwargs):
    path = kwargs.get('path', None)
    device = kwargs.get('device', None)
    scaleFactor = kwargs.get('scale', 1)

    verts, faces, aux = load_obj(path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    verts = verts * scaleFactor

    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

    return trg_mesh

if __name__ == '__main__':
    dev = torch.device("cuda:0")
    dev_1 = torch.device("cuda:1")
    torch.cuda.empty_cache()
    path = 'bunny.obj'

    numPoints = 500

    BS = 40
    thetaStart = 0
    thetaStop = 359
    thetaStep = 1
    rStart = 1
    rStop = 1
    zStart = 1
    zStop = 1.5
    zStep = .001
    sceneDimX = [-.4, .4]
    sceneDimY = [-.4, .4]
    sceneDimZ = [-.4, .4]
    pixDim = [255, 255, 255]
    propLoss = True

    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)

    fig1 = plt.figure(figsize=(5, 5))
    ax1 = Axes3D(fig1)

    with torch.no_grad():
        ###################### Ground Truth ###########################
        torch.cuda.set_device(dev)

        trg_mesh = getMesh(path=path, device=dev, scale=0.3)

        #plot_pointcloud(trg_mesh)

        RP_GT = RenderParameters(device=dev)
        RP_GT.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep,
                                 rStart=rStart, rStop=rStop, zStart=zStart, zStop=zStop)
        RP_GT.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_GT.generateTransmitSignal(compute=True)

        #ps_GT = sample_points_from_meshes(trg_mesh, numPoints).squeeze()

        #simulateWaveformsBatched(RP_GT, ps_GT, propLoss=True)

        BF_GT = Beamformer(RP=RP_GT)
        #BF_GT.Beamform(RP_GT, BI=range(0, RP_GT.numProj), soft=True)
        #BF_GT.display2Dscene(path='pics6/GT.png', show=False)

        ######################## Estimated ###############################
        torch.cuda.set_device(dev_1)
        #src_mesh = getMesh(path=path, device=dev_1, scale=.01)
        src_mesh = ico_sphere(4, dev_1)
        src_mesh = src_mesh.scale_verts(0.2)

        #plot_pointcloud(src_mesh)

        RP_EST = RenderParameters(device=dev_1)
        RP_EST.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                  rStop=rStop, zStart=zStart, zStop=zStop)
        RP_EST.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        RP_EST.generateTransmitSignal(compute=True)

        BF_EST = Beamformer(RP=RP_EST)


        ######################################################################

        ############Setup stuff ######################################



        vis = visdom.Visdom()
        loss_window = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch', ylabel='Loss', title='L1 Loss', legend=['Loss']))
        wiggle = 1.0 * (1 / RP_EST.Fs) * RP_EST.c
        noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))
        epochs = 100000
        w_L1 = 1.0
        w_wass = 1000.0
        w_edge = 1.0
        w_normal = 0.1
        w_laplacian = 0.1

        plot_period = 10

        l1_losses = []
        laplacian_losses = []
        edge_losses = []
        normal_losses = []

        Projs = range(0, RP_EST.numProj)
        batches = list(batchFromList(Projs, BS))
        #################################################################

    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=dev_1, requires_grad=True)
    print(deform_verts.shape)
    wass1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.01, diameter=1.0)
    optimizer = torch.optim.Adam([deform_verts], lr=.001)
    lr = .01
    numberZs = 20



    for i in range(0, epochs):
        #for j in range(0, numberZs):
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)

        full_batch = random.sample(Projs, k=BS)
        batches = list(batchFromList(full_batch, 60))

        losses = []

        pixels = np.random.choice(RP_EST.numPix3D, size=18000, replace=False)
        randPixels = RP_EST.pixPos3D[pixels, :]

        for batch in batches:
            torch.cuda.set_device(dev)
            sample_trg = sample_points_from_meshes(trg_mesh, numPoints).squeeze().to(dev)
            torch.cuda.set_device(dev_1)
            sample_src = sample_points_from_meshes(new_src_mesh, numPoints).squeeze().to(dev_1)

            with torch.no_grad():
                simulateWaveformsBatched(RP_GT, sample_trg, batch, propLoss=True)
                GT = BF_GT.Beamform(RP_GT, BI=range(0, len(batch)), soft=True, pixels=randPixels).abs().to(dev_1)
            ################## Calculate L1 Loss between beamforme images ###################
            #z = np.random.choice(RP_EST.zVect)

            simulateWaveformsBatched(RP_EST, sample_src, batch, propLoss=True)
            est = BF_EST.Beamform(RP_EST, BI=range(0, len(batch)), soft=True, pixels=randPixels).abs().to(dev_1)

            #est_norm = est / torch.norm(est, p=2)
            #GT_norm = GT / torch.norm(GT, p=2)

            est_pdf = est/torch.sum(est)
            GT_pdf = GT/torch.sum(GT)

            #pixels = torch.from_numpy(randPixels).to(dev_1)
            pixels = BF_EST.pixels.type(torch.float64)
            pixels = pixels.to(dev_1)

            wass_loss = w_wass * wass1_loss(est_pdf.unsqueeze(1), pixels, GT_pdf.unsqueeze(1), pixels)

            #cosine_loss = 1 - torch.nn.functional.cosine_similarity(est_norm.unsqueeze(0), GT_norm.unsqueeze(0))

            #L1 = w_L1 * cosine_loss
            #L1 = w_L1 * wass_loss

            # and (b) the edge length of the predicted mesh
            #loss_edge = w_edge * mesh_edge_loss(new_src_mesh)

            # mesh normal consistency
            #loss_normal = w_normal * mesh_normal_consistency(new_src_mesh)

            # mesh laplacian smoothing
            #loss_laplacian = w_laplacian * mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            loss = wass_loss
            losses.append(loss)

            #print('Total loss: ' + str(loss.data))

            loss.backward()

        #if i % plot_period == 0:
        #optimizer.step()
        #if i % 10 == 0:
        #if i < 200:
        deform_verts.data += noise.sample(deform_verts.shape).squeeze().to(dev_1)

        optimizer.step()

        #deform_verts.data -= lr*deform_verts.grad
        #deform_verts.grad.data.zero_()

        torch.cuda.set_device(dev)
        loss_chamfer, _ = chamfer_distance(sample_trg.unsqueeze(0), sample_src.unsqueeze(0).to(dev))
        loss_chamfer = loss_chamfer*100

        total_loss = torch.sum(torch.stack(losses))

        losses.clear()

        vis.line(X=torch.ones((1)).cpu() * i, Y=total_loss.unsqueeze(0).cpu(), win=loss_window, name='Wass Loss', update='append')
        vis.line(X=torch.ones((1)).cpu() * i, Y=loss_chamfer.unsqueeze(0).cpu(), win=loss_window, name='Chamfer Loss', update='append')
        #vis.line(X=torch.ones((1)).cpu() * i, Y=loss_normal.unsqueeze(0).cpu(), win=loss_window, name='loss_normal',update='append')
        #vis.line(X=torch.ones((1)).cpu() * i, Y=loss_laplacian.unsqueeze(0).cpu(), win=loss_window, name='loss_laplacian', update='append')

        #BF_GT.display2Dscene(show=True)

        torch.cuda.set_device(dev_1)
        plot_pointcloud(new_src_mesh, fig, ax, (i * 20) % 360, title="iter: %d" % i)
        torch.cuda.set_device(dev)
        plot_pointcloud(trg_mesh, fig1, ax1, (i * 20) % 360, title="GT")









