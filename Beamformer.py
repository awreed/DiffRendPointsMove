import numpy as np
from RenderParameters import RenderParameters
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import torch
from utils import *
import scipy

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
import time
from Complex import *


class Beamformer:
    def __init__(self, **kwargs):
        self.RP = kwargs.get('RP', None)
        self.dev = self.RP.dev

        #self.window = self.linearWindow(self.RP)
        #self.window = self.sincWindow(self.RP)
        #self.window = self.heavySideStep()
        #self.window = self.gaussianWindow(self.RP)
        self.window = self.linearStepWindow(1)
        #self.window = self.linearWindow()

        #Define scene parameters set to None and initialed by initScene()
        self.scene = None
        self.sceneDimX = None
        self.sceneDimY = None
        self.sceneDimZ = None
        self.nPix = None
        self.xVect = None
        self.yVect = None
        self.zVect = None
        self.numPix = None
        self.sceneCenter = None
        self.pixPos = None

        # Forward kwargs to initScene()
        self.initScene(sceneDimX=kwargs.get('sceneDimX'), sceneDimY=kwargs.get('sceneDimY'),
                       sceneDimZ=kwargs.get('sceneDimZ'), nPix=kwargs.get('nPix'))

    def initScene(self, **kwargs):
        self.sceneDimX = kwargs.get('sceneDimX', np.array([-.15, .15]))
        self.sceneDimY = kwargs.get('sceneDimY', np.array([-.15, .15]))
        self.sceneDimZ = kwargs.get('sceneDimZ', np.array([-.15, .15]))

        self.nPix = kwargs.get('nPix', np.array([128, 128, 64]))

        self.xVect = np.linspace(self.sceneDimX[0], self.sceneDimX[1], self.nPix[0])
        self.yVect = np.linspace(self.sceneDimY[0], self.sceneDimY[1], self.nPix[1])
        self.zVect = np.linspace(self.sceneDimZ[0], self.sceneDimZ[1], self.nPix[2])

        self.numPix = np.size(self.xVect) * np.size(self.yVect) * np.size(self.zVect)
        self.sceneCenter = np.array([np.median(self.xVect), np.median(self.yVect), np.median(self.zVect)])
        (x, y, z) = np.meshgrid(self.xVect, self.yVect, self.zVect)
        pixPos = np.hstack(
            (np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)), np.reshape(z, (np.size(z), 1))))
        # Convert pixel positions to tensor
        self.pixPos = torch.from_numpy(pixPos).to(self.dev)
        self.pixPos.requires_grad = False


    def heavySideStep(self):
        full_window = []
        w = 10
        for ind in range(0, self.RP.nSamples):
            left = torch.ones(ind)
            right = torch.ones(int(self.RP.nSamples-ind))
            right[0] = w
            window = torch.cat((left, right), 0)
            #plt.stem(window.detach().cpu().numpy(), use_line_collection=True)
            #plt.show()
            full_window.append(window)

        window = torch.stack(full_window).to(self.RP.dev)
        return window

    def linearStepWindow(self, w):
        full_window = []
        for ind in range(0, self.RP.nSamples):
            left = torch.linspace(0, 1, ind)
            right = torch.linspace(1, 0, int(self.RP.nSamples-ind))
            left[0:-1] = left[0:-1]*w
            right[1:] = right[1:]*w
            window = torch.cat((left, right), 0)
            full_window.append(window)

        window = torch.stack(full_window).to(self.RP.dev)
        #plt.clf()
        #plt.stem(window[1000, :].detach().cpu().numpy(), use_line_collection=True)
        #plt.show()

        return window

    # Pre-compute matrix of soft index values for
    def linearWindow(self):
        full_window = []

        for ind in range(0, self.RP.nSamples):
            windowLeft = torch.linspace(0, 1, ind)
            windowRight = torch.linspace(1, 0, int(self.RP.nSamples-ind))
            window = torch.cat((windowLeft, windowRight), 0)
            full_window.append(window)

        window = torch.stack(full_window).to(self.RP.dev)

        return window

    def sincWindow(self, RP):
        full_window = []
        u = torch.linspace(0, RP.nSamples-1, RP.nSamples).detach().cpu().numpy()
        for i in range(0, len(u)):
            window = torch.from_numpy(np.sinc(u[i] - u))
            full_window.append(window)

        return torch.stack(full_window).to(RP.dev)

    def gaussianWindow(self, RP):
        full_window = []
        mu, sigma = 1000, 100
        x_values = np.arange(0, RP.nSamples, 1)
        for i in range(0, len(x_values)):
            y_values = scipy.stats.norm(i, sigma)
            vals = torch.from_numpy(y_values.pdf(x_values))
            full_window.append(vals)
        return torch.stack(full_window).to(RP.dev)
        #plt.clf()
        #plt.stem(x_values, y_values.pdf(x_values), use_line_collection=True)
        #plt.show()

    # 2D/3D Beamformer with option for soft indexing so that its differentiable
    def Beamformer(self, RP, BI = None, soft=True):
        posVecList = []
        for i in BI:
            posVecList.append(RP.projDataArray[i].projPos)
        posVec = torch.stack(posVecList).to(RP.dev)

        projWfmList = []
        for i in BI:
            projWfmList.append(RP.projDataArray[i].wfmRC.vector())
        wfmData = torch.stack(projWfmList).to(RP.dev)

        pixGridReal = []
        pixGridImag = []

        #real = wfmData[0, :, 0].detach().cpu().numpy()
        #imag = wfmData[0, :, 1].detach().cpu().numpy()
        #sinc = self.window[900, :].detach().cpu().numpy()
        #plt.stem(sinc, use_line_collection=True)
        #plt.show()

        x = torch.ones(self.numPix, 3).to(RP.dev)

        # Delay and sum where indices selected by multiplying by window - differentiable.
        if soft == True:
            for i in range(0, len(BI)):
                posVec_pix = x * posVec[i, :]
                sum = torch.sum((self.pixPos - posVec_pix) ** 2, 1)
                tofs = 2 * torch.sqrt(sum)

                tof_ind = ((tofs / torch.tensor(RP.c)) * torch.tensor(RP.Fs)).type(torch.long)

                # Multiply by window to index particular time
                real = torch.sum(wfmData[i, :, 0] * self.window[tof_ind, :], dim=1)
                imag = torch.sum(wfmData[i, :, 1] * self.window[tof_ind, :], dim=1)

                pixGridReal.append(real)
                pixGridImag.append(imag)
        # Delay and sum where indices selected by sampling directly - not differentiable.
        else:
            for i in range(0, len(BI)):
                posVec_pix = x * posVec[i, :]
                sum = torch.sum((self.pixPos - posVec_pix) ** 2, 1)
                tofs = 2 * torch.sqrt(sum)

                tof_ind = ((tofs / torch.tensor(RP.c)) * torch.tensor(RP.Fs)).type(torch.long)

                # Select index directly, not differentiable
                real = wfmData[i, tof_ind, 0]
                imag = wfmData[i, tof_ind, 1]

                pixGridReal.append(real)
                pixGridImag.append(imag)

        real_full = torch.sum(torch.stack(pixGridReal), 0)
        imag_full = torch.sum(torch.stack(pixGridImag), 0)

        self.scene = Complex(real=real_full, imag=imag_full)
        return self.scene

    def displayScene(self, **kwargs):
        dim = kwargs.get('dim', 3)
        real = kwargs.get('real', False)
        imag = kwargs.get('imag', False)
        mag = kwargs.get('mag', True)

        sceneReal = self.scene.real.detach().cpu().numpy()
        sceneImag = self.scene.imag.detach().cpu().numpy()
        sceneMag = self.scene.abs().detach().cpu().numpy()

        pixels = self.pixPos.detach().cpu().numpy()

        if dim == 3:
            if mag == True:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=sceneMag, alpha=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()



