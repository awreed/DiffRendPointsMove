from Beamformer import *
from geomloss import SamplesLoss
from pytorch3d.loss import chamfer_distance
from batch_time_delay import *
import visdom
import sys
from utils import *
import random
from Wavefront import *
np.set_printoptions(threshold=sys.maxsize)

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.io import load_obj, save_obj

np.random.seed(0)
random.seed(0)

def plot_pointcloud(mesh, fig, ax, angle, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 500)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    ax.clear()
    ax.scatter3D(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim(-.4, .4)
    ax.set_ylim(-.4, .4)
    ax.set_zlim(-.4, .4)
    ax.set_title(title)
    ax.view_init(190, angle)
    plt.savefig('pics6/' + title + '.png')
    #plt.pause(.05)

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
    if torch.cuda.is_available():
        dev = "cuda:0"
        dev_1 = "cuda:1"
        torch.cuda.empty_cache()
    else:
        print("Fix ur cuda loser")
        dev = "cpu"
    path = 'bunny_simple.obj'

    BS = 180

    thetaStart = 0
    thetaStop = 359
    thetaStep = 1
    rStart = 1
    rStop = 1
    zStart = 1
    zStop = 1
    zStep = .01
    sceneDimX = [-.4, .4]  # min dist needs to equal max dist
    sceneDimY = [-.4, .4]
    sceneDimZ = [0, 0]
    pixDim = [128, 128, 1]
    propLoss = False
    compute = False
    show = False

    with torch.no_grad():
        #torch.cuda.set_device(dev)
        #trg_mesh = getMesh(path=path, device=dev, scale=0.3)

        #src_mesh = ico_sphere(4, dev)
        #src_mesh = src_mesh.scale_verts(0.3)
        # Define objects in scene
        #ps_GT = torch.tensor([[-.3, -.3, 0]], requires_grad=False).to(dev)
        #ps_EST = torch.tensor([[.3, 0.3, 0]], requires_grad=False).to(dev)


        GT_Mesh = ObjLoader('cube_64_bottom_z.obj')
        GT_Mesh.vertices = GT_Mesh.vertices * 0.2
        GT_Mesh.getCentroids()
        EST_Mesh = ObjLoader('cube_64_bottom_z.obj')
        EST_Mesh.vertices = (EST_Mesh.vertices * 0.3)
        EST_Mesh.getCentroids()
        GT = GT_Mesh.centroids

        EST = EST_Mesh.centroids
        #EST[:, 0] = EST[:, 0] * -1
        #EST[:, 1] = EST[:, 1] * -1

        ps_GT = torch.from_numpy(GT).to(dev)
        ps_EST = torch.from_numpy(EST).to(dev)


        # Define the scene
        RP = RenderParameters(device=dev)
        RP.defineProjectorPos(thetaStart=thetaStart, thetaStop=thetaStop, thetaStep=thetaStep, rStart=rStart,
                                 rStop=rStop, zStart=zStart, zStop=zStop, zStep=zStep)
        RP.defineSceneDimensions(sceneDimX=sceneDimX, sceneDimY=sceneDimY, sceneDimZ=sceneDimZ, pixDim=pixDim)
        #RP.generateTransmitSignal(compute=compute)
        RP.generateImpulse()

        # Define a beamformer
        BF = Beamformer(RP=RP)

        vis = visdom.Visdom()

        loss_window_small = vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iter', ylabel='Loss', title='Loss', legend=['']))

    ps_est = ps_EST.clone()
    #deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=dev, requires_grad=True)
    ps_est.requires_grad = True
    lr = 1E3
    cs = torch.nn.CosineSimilarity()
    optimizer = torch.optim.SGD([ps_est], lr=lr)

    wiggle = 2.0 * (1 / RP.Fs) * RP.c
    noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([wiggle]))
    numPoints = 500

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111, projection='3d')

    for i in range(1, 100000):
        print("I is " + str(i))
        optimizer.zero_grad()

        #new_src_mesh = src_mesh.offset_verts(deform_verts)

        # Sample random batch of projectors
        batch = random.sample(range(0, RP.numProj), k=BS)

        #sample_trg = sample_points_from_meshes(trg_mesh, numPoints).squeeze().to(dev)
        #sample_src = sample_points_from_meshes(new_src_mesh, numPoints).squeeze().to(dev)

        # Beamform the GT image
        with torch.no_grad():
            simulateSASWaveformsPointSource(RP, ps_GT, BI=batch)
            #simulateWaveformsBatched(RP, ps_GT, batch, propLoss=propLoss)
            GT = BF.Beamform(RP, BI=range(0, len(batch)), soft=False, pixels=RP.pixPos3D).to(dev_1)

        # Beamform the estimate image
        #simulateWaveformsBatched(RP, ps_est, batch, propLoss=propLoss)
        simulateSASWaveformsPointSource(RP, ps_est, BI=batch)

        est = BF.Beamform(RP, BI=range(0, len(batch)), soft=False, pixels=RP.pixPos3D).to(dev_1)

        est = est/(torch.sum(est).detach())
        GT = GT/(torch.sum(GT).detach())

        #est = (est - est.min().detach())/(est.max().detach() - est.min().detach())
        #GT = (GT - GT.min().detach()) / (GT.max().detach() - GT.min().detach())
        est.retain_grad()

        BF.display2Dscene(path='test_pics/GT' + str(i) + '.png', scene=GT)
        BF.display2Dscene(path='test_pics/est' + str(i) + '.png', scene=est)

        diff = (GT - est)**2

        loss = torch.sqrt(torch.sum(diff))
        #loss = 1 - cs(GT.unsqueeze(0), est.unsqueeze(0))

        #loss1 = torch.norm(GT-est, p=2)
        #print(loss)
        #print(loss1)
        #loss = torch.sum(pixel_loss)

        #loss = 1 - cs(GT.unsqueeze(0), est.unsqueeze(0))
        loss.backward()

        #grad_img = (est.grad - est.grad.min())/(est.grad.max() - est.grad.min())
        #grad_img1 = (est_norm.grad - est_norm.grad.min()) / (est_norm.grad.max() - est_norm.grad.min())

        #print(torch.min(grad_img), torch.max(grad_img))
        BF.display2Dscene(path='test_pics/grad' + str(i) + '.png', scene=est.grad)
        #print(diff.shape)
        BF.display2Dscene(path='test_pics/l2_diff' + str(i) + '.png', scene=diff)

        loss_chamfer, _ = chamfer_distance(ps_GT.unsqueeze(0), ps_est.unsqueeze(0).to(dev))
        loss_chamfer = loss_chamfer * 1000

        #print(deform_verts.grad * lr)
        #minC = -1/(lr*10)
        #maxC = 1/(lr*10)
        #print(minC)
        #print(maxC)
        #ps_est.grad.data.clamp_(minC, maxC)

        #optimizer.step()


        #plotGT = RP.data['GTWfm'].squeeze().detach().cpu().numpy()
        #plotEST = RP.data['ESTWfm'].squeeze().detach().cpu().numpy()
        #plotESTGrad = RP.data['ESTGrad'].squeeze().detach().cpu().numpy()
        #plotSigGrad = RP.data['indSig'].squeeze().detach().cpu().numpy()
        #plotpDataWfm = RP.data['pDataWfm'].squeeze().detach().cpu().numpy()
        #plotPR = RP.data['arg'].squeeze().detach().cpu().numpy()
        #plotReal = RP.data['real']

        #plt.clf()
        #plt.stem(plotGT, use_line_collection=True)
        #plt.savefig('test_pics/GTwfm' + str(i) + '.png')

        #plt.clf()
        #plt.stem(plotEST, use_line_collection=True)
        #plt.savefig('test_pics/ESTwfm' + str(i) + '.png')

        #plt.clf()
        #plt.stem(plotESTGrad, use_line_collection=True)
        #plt.savefig('test_pics/ESTwfmGrad' + str(i) + '.png')

        #plt.clf()
        #plt.stem(plotSigGrad, use_line_collection=True)
        #plt.savefig('test_pics/indSig' + str(i) + '.png')

        #plt.clf()
        #plt.stem(plotpDataWfm, use_line_collection=True)
        #plt.savefig('test_pics/pDataWfm' + str(i) + '.png')

        #plt.clf()
        #plt.stem(plotPR, use_line_collection=True)
        #plt.savefig('test_pics/pr' + str(i) + '.png')

        #BF.display2Dscene(path='test_pics/real' + str(i) + '.png', scene=plotReal)

        RP.freeHooks()
        ps_est.grad[:, 2] = 0
        #print(ps_est.grad)
        print("Point gradient")
        print(ps_est.grad)
        #print("Point Position")
        #print(ps_est)
        #ps_est.grad[:, 2] = 0
        #ps_est.data[:, 0:3] -= lr * ps_est.grad[:, 0:3]
        #ps_est.grad.data.zero_()
        optimizer.step()

        #if i % 25 == 0:
        #    for param_group in optimizer.param_groups:
        #        lr = lr * 2
        #        param_group['lr'] = lr
        #print(ps_est.data)
        #ps_est.data[:, 0:2] += noise.sample(ps_est.shape).squeeze().to(dev)[:, 0:2]

        #ps_numpy = ps_est.data.detach().cpu().numpy()

        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window_small, name = 'Image Loss', update='append')
        vis.line(X=torch.ones((1)).cpu() * i, Y=loss_chamfer.unsqueeze(0).cpu(), win=loss_window_small, name='Point to Point Loss',
                 update='append')

        torch.cuda.set_device(dev)
        #plot_pointcloud(new_src_mesh, fig, ax, (i * 20) % 360, title="iter: %d" % i)
        #torch.cuda.set_device(dev)
        #plot_pointcloud(trg_mesh, fig1, ax1, (i * 20) % 360, title="GT")

        #ax1.clear()
        #ax1.scatter(ps_numpy[:, 0], ps_numpy[:, 1], ps_numpy[:, 2])
        #ax1.set_xlim3d((-.4, .4))
        #ax1.set_ylim3d((-.4, .4))
        #ax1.set_zlim3d((-.4, .4))
        #ax1.set_xlabel('X')
        #ax1.set_ylabel('Y')
        #ax1.set_zlabel('Z')
        #plt.pause(.05)










