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

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import time
from Complex import *
import scipy.misc
from PIL import Image


class Beamformer:
    def __init__(self, **kwargs):
        self.RP = kwargs.get('RP', None)
        self.dev = self.RP.dev

        # self.window = self.linearWindow(self.RP)
        # self.window = self.sincWindow(self.RP)
        # self.window = self.heavySideStep()
        # self.window = self.gaussianWindow(self.RP)
        self.window = self.linearStepWindow(1)
        # self.window = self.expWindow()

        self.scene = None

        self.sceneDimX = self.RP.sceneDimX
        self.sceneDimY = self.RP.sceneDimY
        self.sceneDimZ = self.RP.sceneDimZ

        # self.nPix = self.RP.pixDim

        self.xVect = self.RP.xVect
        self.yVect = self.RP.yVect
        self.zVect = self.RP.zVect

        self.numPix = self.RP.numPix
        # self.sceneCenter = self.RP.sceneCenter
        self.pixPos = self.RP.pixPos
        # Convert pixel positions to tensor
        self.pixPos = torch.from_numpy(self.pixPos)
        self.pixPos = self.pixPos.type(torch.float64).to(self.dev)
        self.pixPos.requires_grad = False
        self.pixels = None

    def heavySideStep(self):
        full_window = []
        w = 10
        for ind in range(0, self.RP.nSamples):
            left = torch.ones(ind)
            right = torch.ones(int(self.RP.nSamples - ind))
            right[0] = w
            window = torch.cat((left, right), 0)
            # plt.stem(window.detach().cpu().numpy(), use_line_collection=True)
            # plt.show()
            full_window.append(window)

        window = torch.stack(full_window).to(self.RP.dev)
        return window

    def linearStepWindow(self, w):
        full_window = []
        for ind in range(0, self.RP.nSamples):
            left = torch.linspace(0, 1, ind)
            right = torch.linspace(1, 0, int(self.RP.nSamples - ind))
            # left[0:-1] = left[0:-1]*w
            # right[1:] = right[1:]*w
            window = torch.cat((left, right), 0)
            full_window.append(window)

        window = torch.stack(full_window).to(self.RP.dev)
        # plt.clf()
        # plt.stem(window[1000, :].detach().cpu().numpy(), use_line_collection=True)
        # plt.show()

        return window

    # Pre-compute matrix of soft index values for
    def expWindow(self):
        full_window = []
        sigma = 6

        for ind in range(0, self.RP.nSamples):
            left = torch.linspace(0, 1, ind)
            right = torch.linspace(1, 0, int(self.RP.nSamples - ind))
            windowLeft = torch.exp(sigma * left) - torch.ones_like(left)
            windowRight = torch.exp(sigma * right) - torch.ones_like(right)
            window = torch.cat((windowLeft, windowRight), 0)
            window = window / torch.max(window)
            full_window.append(window)

        # plt.clf()
        # plt.stem(full_window[500].detach().cpu().numpy())
        # plt.show()

        window = torch.stack(full_window).to(self.RP.dev)

        return window

    def sincWindow(self, RP):
        full_window = []
        u = torch.linspace(0, RP.nSamples - 1, RP.nSamples).detach().cpu().numpy()
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
        # plt.clf()
        # plt.stem(x_values, y_values.pdf(x_values), use_line_collection=True)
        # plt.show()

    # 2D/3D Beamformer with option for soft indexing so that its differentiable
    def Beamform(self, RP, BI=None, soft=True, **kwargs):
        posVecList = []
        projWfmList = []

        for i in BI:
            posVecList.append(RP.projDataArray[i].projPos)
            projWfmList.append(RP.projDataArray[i].wfm)
        posVec = torch.stack(posVecList).to(RP.dev)
        wfmData = torch.stack(projWfmList).to(RP.dev)

        #if not wfmData.requires_grad:
        #    RP.save(key='GTWfm', val=wfmData)
        #if wfmData.requires_grad:
        #    RP.save(key='ESTWfm', val=wfmData)
        #    h = wfmData.register_hook(lambda x: RP.save(key='ESTGrad', val=x))
        #    RP.hooks.append(h)

        pixGridReal = []
        pixGridImag = []

        z = kwargs.get('z', None)

        pixels = kwargs.get('pixels', None)

        if pixels is not None:
            pixels = torch.from_numpy(pixels).to(self.dev)
            pixels.requires_grad = False

        if pixels is None and z is not None:
            # pixels = self.pixPos
            zTorch = (torch.ones(self.numPix) * z).unsqueeze(1).to(self.dev)
            zTorch = zTorch.type(torch.float64)

            # pixels = torch.cat((zTorch, self.pixPos), 1)
            pixels = torch.cat((self.pixPos, zTorch), 1)
            pixels.requires_grad = False

        self.pixels = pixels

        x = torch.ones_like(pixels).to(RP.dev)
        # Delay and sum where indices selected by multiplying by window - differentiable.
        if soft == True:
            for i in range(0, len(BI)):
                posVec_pix = x * posVec[i, :]
                # print(posVec_pix.dtype)
                dist = torch.sum((pixels - posVec_pix) ** 2, 1)
                tofs = (2 * torch.sqrt(dist)) - self.RP.minDist
                # print(tofs.dtype)

                tof_ind = ((tofs / torch.tensor(RP.c)) * torch.tensor(RP.Fs)).type(torch.long)

                # Multiply by window to index particular time
                real = torch.sum(wfmData[i, :] * self.window[tof_ind, :], dim=1)

                # imag = torch.sum(wfmData[i, :] * self.window[tof_ind, :], dim=1)

                # print(real.dtype)
                # print(imag.dtype)

                pixGridReal.append(real)
                # pixGridImag.append(imag)
        # Delay and sum where indices selected by sampling directly - not differentiable.
        else:
            for i in range(0, len(BI)):
                posVec_pix = x * posVec[i, :]
                dist = torch.sum((pixels - posVec_pix) ** 2, 1)
                tofs = (2 * torch.sqrt(dist)) - self.RP.minDist

                tof_ind = ((tofs / torch.tensor(RP.c)) * torch.tensor(RP.Fs)).type(torch.long)

                # Select index directly, not differentiable
                real = wfmData[i, tof_ind]

                if real.requires_grad:
                    h = real.register_hook(lambda x: RP.save(key='real', val=x))
                    RP.hooks.append(h)

                # imag = wfmData[i, tof_ind, 1]

                pixGridReal.append(real)
                # pixGridImag.append(imag)

        real_full = torch.sum(torch.stack(pixGridReal), 0)
        # imag_full = torch.sum(torch.stack(pixGridImag), 0)

        # self.scene = Complex(real=real_full, imag=imag_full)
        self.scene = real_full.to(RP.dev)
        return self.scene

    def sideByside(self, **kwargs):
        img1 = kwargs.get('img1', None)
        img2 = kwargs.get('img2', None)
        img3 = kwargs.get('img3', None)
        path = kwargs.get('path', None)
        show = kwargs.get('show', None)

        x_vals = torch.unique(self.pixPos[:, 0]).numel()
        y_vals = torch.unique(self.pixPos[:, 1]).numel()

        img1 = img1.view(x_vals, y_vals)
        img2 = img2.view(x_vals, y_vals)
        img3 = img3.view(x_vals, y_vals)

        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
        img3 = img3.detach().cpu().numpy()

        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
        img3 = (img3 - np.min(img3)) / (np.max(img3) - np.min(img3))

        combined = np.hstack((img1, img2, img3))

        plt.imshow(combined)

        # if show:
        #    plt.pause(.05)
        if path:
            plt.savefig(path)
            plt.clf()

    def display2Dscene(self, **kwargs):
        path = kwargs.get('path', None)
        show = kwargs.get('show', False)
        scene = kwargs.get('scene', None)

        x_vals = torch.unique(self.pixels[:, 0]).numel()
        y_vals = torch.unique(self.pixels[:, 1]).numel()

        sceneXY = scene.view(x_vals, y_vals)

        # sceneXY = self.scene.abs().view(x_vals, y_vals)
        plt.clf()
        plt.imshow(sceneXY.detach().cpu().numpy())
        plt.colorbar()

        if path is not None:
            plt.savefig(path)
        if show is True:
            plt.show()

    def displayScene(self, **kwargs):
        dim = kwargs.get('dim', 3)
        real = kwargs.get('real', False)
        imag = kwargs.get('imag', False)
        mag = kwargs.get('mag', True)
        ax = kwargs.get('ax', None)

        sceneReal = self.scene.real.detach().cpu().numpy()
        sceneImag = self.scene.imag.detach().cpu().numpy()
        sceneMag = self.scene.abs().detach().cpu().numpy()

        pixels = self.pixels.detach().cpu().numpy()
        u = np.mean(sceneMag)
        std = np.std(sceneMag)
        # print(u)
        # print(std)
        w = 1
        sceneMag[sceneMag[:] < u + w * std] = np.nan
        # print(np.count_nonzero(~np.isnan(sceneMag)))

        if dim == 3:
            if mag == True:
                ax.clear()
                ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=sceneMag, alpha=0.5)
                ax.set_xlim3d((-.4, .4))
                ax.set_ylim3d((-.4, .4))
                ax.set_zlim3d((-.4, .4))
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                plt.pause(.05)
