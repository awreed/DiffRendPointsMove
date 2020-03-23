import numpy as np
from RenderParameters import RenderParameters
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import multiprocessing
import torch
from utils import *

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
import time
from Complex import *


class Beamformer:
    def __init__(self, **kwargs):
        self.sceneDimX = kwargs.get('sceneDimX', np.array([-.15, .15]))
        self.sceneDimY = kwargs.get('sceneDimY', np.array([-.15, .15]))
        self.sceneDimZ = kwargs.get('sceneDimZ', np.array([-.15, .15]))

        self.RP = kwargs.get('RP', None)

        self.dev = self.RP.dev

        self.nPix = kwargs.get('nPix', np.array([128, 128, 64]))

        self.xVect = np.linspace(self.sceneDimX[0], self.sceneDimX[1], self.nPix[0])
        self.yVect = np.linspace(self.sceneDimY[0], self.sceneDimY[1], self.nPix[1])
        self.zVect = np.linspace(self.sceneDimZ[0], self.sceneDimZ[1], self.nPix[2])

        self.window = self.windowIndex(self.RP)

        self.dim = kwargs.get('dim', 3)

        self.scene = None

        if self.dim == 3:
            self.numPix = np.size(self.xVect) * np.size(self.yVect) * np.size(self.zVect)
            self.sceneCenter = np.array([np.median(self.xVect), np.median(self.yVect), np.median(self.zVect)])
            (x, y, z) = np.meshgrid(self.xVect, self.yVect, self.zVect)
            pixPos = np.hstack(
                (np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)), np.reshape(z, (np.size(z), 1))))
            # Convert pixel positions to tensor
            self.pixPos = torch.from_numpy(pixPos).to(self.dev)
            self.pixPos.requires_grad = True
        else:
            self.numPix = np.size(self.xVect) * np.size(self.yVect)
            self.sceneCenter = np.array([np.median(self.xVect), np.median(self.yVect), np.median(self.zVect)])
            (x, y) = np.meshgrid(self.xVect, self.yVect)
            pixPos = np.hstack(
                (np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1))))
            # Convert pixel positions to tensor
            self.pixPos = torch.from_numpy(pixPos).to(self.dev)
            self.pixPos.requires_grad = True

    # Pre-compute matrix of soft index values for
    def windowIndex(self, RP):
        full_window = []
        p=1#shape the window to immitate delta sampling with ahigh p value
        for ind in range(0, RP.nSamples):
            windowLeft = torch.linspace(0, 1, ind)**p
            windowRight = torch.linspace(1, 0, int(RP.nSamples-ind))**p
            window = torch.cat((windowLeft, windowRight), 0)
            full_window.append(window)

        return torch.stack(full_window).to(RP.dev)

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
