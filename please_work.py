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
    sceneDimX = [-.4, .4]  # min dist needs to equal max dist
    sceneDimY = [-.4, .4]
    sceneDimZ = [-.4, .4]
    pixDim = [128, 128, 64]
    propLoss = False
    compute = False
    show = False

    with torch.no_grad():
        # Define objects in scene
        ps_GT = torch.tensor([[-.3, -.3, -.3], [-.3, -.3, -.3]], requires_grad=False).to(dev)
        ps_EST = torch.tensor([[.3, 0.3, .3], [.3, .3, .3]], requires_grad=False).to(dev)


        #GT_Mesh = ObjLoader('cube_64_bottom_z.obj')
        #GT_Mesh.vertices = GT_Mesh.vertices * 0.3
        #GT_Mesh.getCentroids()
        #EST_Mesh = ObjLoader('cube_64_bottom_z.obj')
        #EST_Mesh.vertices = EST_Mesh.vertices * 0.01
        #EST_Mesh.getCentroids()

        #GT = GT_Mesh.centroids

        #EST = EST_Mesh.centroids

        #ps_GT = torch.from_numpy(GT).to(dev)
        #ps_EST = torch.from_numpy(EST).to(dev)


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
    ps_est.requires_grad = True
    lr = .001
    cs = torch.nn.CosineSimilarity()
    optimizer = torch.optim.SGD([ps_est], lr=lr)

    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111, projection='3d')

    for i in range(1, 100000):
        optimizer.zero_grad()

        # Sample random batch of projectors
        batch = random.sample(range(0, RP.numProj), k=BS)

        # Beamform the GT image
        with torch.no_grad():
            simulateSASWaveformsPointSource(RP, ps_GT, BI=batch)
            #simulateWaveformsBatched(RP, ps_GT, batch, propLoss=propLoss)
            GT = BF.Beamform(RP, BI=range(0, len(batch)), soft=False, pixels=RP.pixPos3D).to(dev_1)

        # Beamform the estimate image
        #simulateWaveformsBatched(RP, ps_est, batch, propLoss=propLoss)
        simulateSASWaveformsPointSource(RP, ps_est, BI=batch)

        est = BF.Beamform(RP, BI=range(0, len(batch)), soft=False, pixels=RP.pixPos3D).to(dev_1)

        est = (est - est.min().detach())/(est.max().detach() - est.min().detach())
        GT = (GT - GT.min().detach()) / (GT.max().detach() - GT.min().detach())
        est.retain_grad()

        #BF.display2Dscene(path='test_pics/GT' + str(i) + '.png', scene=GT)
        #BF.display2Dscene(path='test_pics/est' + str(i) + '.png', scene=est)

        diff = torch.pow(GT-est, 2)

        loss = torch.sqrt(torch.sum(diff))
        #loss1 = torch.norm(GT-est, p=2)
        #print(loss)
        #print(loss1)
        #loss = torch.sum(pixel_loss)

        #loss = 1 - cs(GT.unsqueeze(0), est.unsqueeze(0))
        loss.backward()

        #grad_img = (est.grad - est.grad.min())/(est.grad.max() - est.grad.min())
        #grad_img1 = (est_norm.grad - est_norm.grad.min()) / (est_norm.grad.max() - est_norm.grad.min())

        #print(torch.min(grad_img), torch.max(grad_img))
        #BF.display2Dscene(path='test_pics/grad' + str(i) + '.png', scene=est.grad)
        #print(diff.shape)
        #BF.display2Dscene(path='test_pics/l2_diff' + str(i) + '.png', scene=diff)

        loss_chamfer, _ = chamfer_distance(ps_GT.unsqueeze(0), ps_est.unsqueeze(0).to(dev))
        loss_chamfer = loss_chamfer * 1000

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
        #ps_est.grad[:, 2] = 0
        #print("Point gradient")
        #print(ps_est.grad)
        #print("Point Position")
        #ps_est.grad[:, 2] = 0
        #ps_est.data[:, 0:3] -= lr * ps_est.grad[:, 0:3]
        #ps_est.grad.data.zero_()
        optimizer.step()
        print(ps_est.data)

        ps_numpy = ps_est.data.detach().cpu().numpy()

        vis.line(X=torch.ones((1)).cpu() * i, Y=loss.unsqueeze(0).cpu(), win=loss_window_small, name = 'Image Loss', update='append')
        vis.line(X=torch.ones((1)).cpu() * i, Y=loss_chamfer.unsqueeze(0).cpu(), win=loss_window_small, name='Point to Point Loss',
                 update='append')

        #ax1.clear()
        #ax1.scatter(ps_numpy[:, 0], ps_numpy[:, 1], ps_numpy[:, 2])
        #ax1.set_xlim3d((-.4, .4))
        #ax1.set_ylim3d((-.4, .4))
        #ax1.set_zlim3d((-.4, .4))
        #ax1.set_xlabel('X')
        #ax1.set_ylabel('Y')
        #ax1.set_zlabel('Z')
        #plt.pause(.05)










