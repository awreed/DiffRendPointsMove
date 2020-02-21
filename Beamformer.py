import numpy as np
from RenderParameters import RenderParameters
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import torch
from utils import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import time
from Complex import *


class Beamformer:
    def __init__(self, **kwargs):
        self.sceneDimX = kwargs.get('sceneDimX', np.array([-.15, .15]))
        self.sceneDimY = kwargs.get('sceneDimY', np.array([-.15, .15]))
        self.sceneDimZ = kwargs.get('sceneDimZ', np.array([-.15, .15]))

        self.nPix = kwargs.get('nPix', np.array([128, 128, 64]))

        self.xVect = np.linspace(self.sceneDimX[0], self.sceneDimX[1], self.nPix[0])
        self.yVect = np.linspace(self.sceneDimY[0], self.sceneDimY[1], self.nPix[1])
        self.zVect = np.linspace(self.sceneDimZ[0], self.sceneDimZ[1], self.nPix[2])

        self.dim = kwargs.get('dim', 3)

        self.scene = None

        if self.dim == 3:
            self.numPix = np.size(self.xVect) * np.size(self.yVect) * np.size(self.zVect)
            self.sceneCenter = np.array([np.median(self.xVect), np.median(self.yVect), np.median(self.zVect)])
            (x, y, z) = np.meshgrid(self.xVect, self.yVect, self.zVect)
            pixPos = np.hstack(
                (np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)), np.reshape(z, (np.size(z), 1))))
            # Convert pixel positions to tensor
            self.pixPos = torch.from_numpy(pixPos).cuda()
            self.pixPos.requires_grad = True
        else:
            self.numPix = np.size(self.xVect) * np.size(self.yVect)
            self.sceneCenter = np.array([np.median(self.xVect), np.median(self.yVect), np.median(self.zVect)])
            (x, y) = np.meshgrid(self.xVect, self.yVect)
            pixPos = np.hstack(
                (np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1))))
            # Convert pixel positions to tensor
            self.pixPos = torch.from_numpy(pixPos).cuda()
            self.pixPos.requires_grad = True

    def beamformTest(self, RP):
        numProj = len(RP.projDataArray)
        posVecList = []
        for i in range(0, numProj):
            posVecList.append(RP.projDataArray[i].projPos)
        posVec = torch.stack(posVecList)

        projWfmList = []
        for i in range(0, numProj):
            projWfmList.append(RP.projDataArray[i].wfmRC.vector())
        wfmData = torch.stack(projWfmList)
       #h = wfmData.register_hook(lambda x: print("wfmData" + str(x[:, :, 0].nonzero().data)))
       # RP.hooks.append(h)

        pixGridReal = []
        pixGridImag = []

        # Much faster now that I loop over projectors rather than pixels
        x = torch.ones(self.numPix, 2)
        z = (torch.ones(self.numPix) * torch.tensor(RP.zs[0]))**2


        for i in range(0, RP.numProj):
            posVec_pix = x * posVec[i, :]
            sum = torch.sum((self.pixPos - posVec_pix)**2, 1)
            #print(sum)
            tot_sum = sum + z
            tofs = 2 * torch.sqrt(tot_sum)
            #tofs = 2 * torch.sqrt(torch.sum(torch.sum((self.pixPos - posVec_pix)**2, 1), (z*torch.tensor([RP.zs[0]]))**2, 1))
            tof_ind = torch.round((tofs / torch.tensor(RP.c)) * torch.tensor(RP.Fs)).long()
            #print(tof_ind)

            pixGridReal.append(wfmData[i, tof_ind.detach(), 0])
            pixGridImag.append(wfmData[i, tof_ind.detach(), 1])

        #self.scene = torch.stack(((torch.sum(torch.stack(pixGridReal), 0)), torch.sum(torch.stack(pixGridImag), 0)), dim=1)
        self.scene = Complex(real=torch.sum(torch.stack(pixGridReal), 0), imag=torch.sum(torch.stack(pixGridImag), 0))
        #h = self.scene.register_hook(lambda x: print("scene" + str(x)))
        #RP.hooks.append(h)
        return self.scene

    def beamform(self, RP):
        numProj = len(RP.projDataArray)

        # Define a vector containing the 3D position of each projector
        #posVec = torch.empty((numProj, 3))
        #for i in range(0, numProj):
        #    posVec.data[i, :] = RP.projDataArray[i].projPos
        posVecList = []
        for i in range(0, numProj):
            posVecList.append(RP.projDataArray[i].projPos)
        posVec = torch.stack(posVecList)

        # Define a vector containing the 3D position of each pixel
        numPix = np.size(self.xVect) * np.size(self.yVect) * np.size(self.zVect)
        (x, y, z) = np.meshgrid(self.xVect, self.yVect, self.zVect)
        pixPos = np.hstack((np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)), np.reshape(z, (np.size(z), 1))))
        # Convert pixel positions to tensor
        pixPos = torch.from_numpy(pixPos)
        # Pack all RC waveforms into matrix
        #wfmData = torch.empty((numProj, int(RP.nSamples), 2), requires_grad=False)
        #for i in range(0, numProj):
        #    wfmData[i, :, :] = RP.projDataArray[i].wfmRC
        projWfmList = []
        for i in range(0, numProj):
            projWfmList.append(RP.projDataArray[i].wfmRC)
        wfmData = torch.stack(projWfmList)
       # print(wfmData.shape)
        #wfmData = np.ndarray.flatten(wfmData)

        # Array to store intensity at each pixel

        pixIntensityReal = []
        pixIntensityImag = []

        #for i in range(0, 1):
        pixel = pixPos[0, :] * torch.ones((numProj, 3))
        tofs = 2 * torch.sqrt(torch.sum((posVec - pixel)**2, 1))
        #print(tofs.shape)
        tof_ind = torch.round((tofs/torch.tensor(RP.c))*torch.tensor(RP.Fs)).long()

        #print(tof_ind.shape)
        #print(RP.projDataArray[0].wfmRC.shape)

        real = torch.sum(wfmData[:, :, 0].gather(1, tof_ind.view(-1, 1)))
        imag = torch.sum(wfmData[:, :, 1].gather(1, tof_ind.view(-1, 1)))

            #pixIntensityReal.append(torch.sum(wfmData[:, :, 0].gather(1, tof_ind.view(-1, 1))))
            #pixIntensityImag.append(torch.sum(wfmData[:, :, 1].gather(1, tof_ind.view(-1, 1))))
        sescene = torch.stack((real, imag)).unsqueeze(0)
        print(self.scene.shape)
        # Save the aggregated pixel intensity for each pixel
        #self.scene = torch.stack((torch.stack(pixIntensityReal), torch.stack(pixIntensityImag)), 1)
        #self.scene = torch.zeros((numPix, 4), dtype=torch.float64)
        self.pixelGrid = pixPos
        #self.scene[:, 3] = compABS(self.pixIntensity)

    def displayScene(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.scene[:, 0].detach().numpy(), self.scene[:, 1].detach().numpy(), self.scene[:, 2].detach().numpy(),
                   c=self.scene[:, 3].detach().numpy())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(self.sceneDimX[0], self.sceneDimX[1])
        ax.set_ylim(self.sceneDimY[0], self.sceneDimY[1])
        ax.set_zlim(self.sceneDimZ[0], self.sceneDimZ[1])

        plt.pause(.05)
        plt.show()