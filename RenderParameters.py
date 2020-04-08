import torch
import numpy as np
from utils import *
import scipy.signal
import math
import matplotlib.pyplot as plt


class RenderParameters:
    def __init__(self, **kwargs):
        # self.Fs = torch.tensor([kwargs.get('Fs', 100000)], requires_grad=True)
        # self.tDur = torch.tensor([kwargs.get('tDur', .02)], requires_grad=True)
        self.Fs = kwargs.get('Fs', 70000) * 1.0
        #self.tDur = kwargs.get('tDur', .02)
        #self.nSamples = int(self.Fs * self.tDur)
        self.nSamples = None # Computed under generateTransmitSignal()

        self.dev = kwargs.get('device', None)

        # Will be used to create torch constant, not differentiable at this time
        self.fStart = kwargs.get('fStart', 30000)  # Chirp start frequency
        self.fStop = kwargs.get('fStop', 10000)  # Chirp stop frequency
        self.tStart = kwargs.get('tStart', 0)  # Chirp start time
        self.tStop = kwargs.get('tStop', .001)  # Chirp stop time
        self.winRatio = kwargs.get('winRatio', 0.1)  # Tukey window ratio for chirp
        self.c = kwargs.get('c', 343.0)

        self.transmitSignal = None  # Transmitted signal (Set to torch type)
        self.pulse = None  # Hilbert transform of the transmitted signal
        self.Pulse = None  # FFT of the hilbert transform of transmitted signal
        self.scene = None  # Stores the processed .obj file
        self.thetaStart = None  # Projector start angle in degrees
        self.thetaStop = None  # Projector stop angle in degrees
        self.thetaStep = None  # Projector step angle in degrees
        self.zStart = None  # Projector start z in meters
        self.zStop = None  # Projector stop z in meters
        self.zStep = None  # Projector step z in meters
        self.rStart = None  # Projector start radius in meters
        self.rStop = None  # Projector stop radius in meters
        self.rStep = None  # Projector step radius in meters
        self.numThetas = None  # Projector number of theta positions
        self.numRs = None  # Projector number of radial positions
        self.numZs = None  # Projector number of z positions
        self.numProj = None  # Number of projectors
        self.projectors = None  # Array of projector 3D coordinates in meters (Set to torch type)
        self.rs = None  # Array of radius values in meters
        self.zs = None  # Array of z values in meters
        self.thetas = None  # Array of theta values in degrees
        self.projDataArray = [] # Accumulate the projectors and associated waveforms before calling backwards()
        self.xStart = None # Grid geometry: Starting x position
        self.xStop = None # Grid geometry: Stopping x position
        self.yStart = None # Grid geometry: Staring y position
        self.yStop = None
        self.zStart = None
        self.zStop = None
        self.numXs = None
        self.numYs = None
        self.numZs = None
        self.xs = None
        self.ys = None
        self.zs = None
        self.xStep = None
        self.yStep = None
        self.hooks = []
        self.sceneDimX = None # Ensonified scene dimensions [-x, x]
        self.sceneDimY = None # Ensonified scene dimensions [-y, y]
        self.sceneDimZ = None # Ensonified scene dimensions [-z, z]
        self.pixDim = None # N pixels in beamformed image in format [x, y, z]
        self.xVect = None # Vector of scene pixels x positions
        self.yVect = None # Vector of scene pixel y positions
        self.zVect = None # Vector of scene pixel z positions
        self.numPix = None # Number of pixels in scene
        self.sceneCenter = None # Center of the ensonified scene
        self.pixPos = None
        self.minDist = None
        self.maxDist = None


    def defineSceneDimensions(self, **kwargs):
        self.sceneDimX = kwargs.get('sceneDimX', [-.4, .4])
        self.sceneDimY = kwargs.get('sceneDimY', [-.4, .4])
        self.sceneDimZ = kwargs.get('sceneDimZ', [0, 0])
        self.pixDim = kwargs.get('pixDim', [128, 128, 1])

        self.xVect = np.linspace(self.sceneDimX[0], self.sceneDimX[1], self.pixDim[0])
        self.yVect = np.linspace(self.sceneDimY[0], self.sceneDimY[1], self.pixDim[1])
        self.zVect = np.linspace(self.sceneDimZ[0], self.sceneDimZ[1], self.pixDim[2])

        self.numPix = np.size(self.xVect) * np.size(self.yVect)
        self.sceneCenter = np.array([np.median(self.xVect), np.median(self.yVect), np.median(self.zVect)])
        (x, y) = np.meshgrid(self.xVect, self.yVect)
        self.pixPos = np.hstack((np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1))))

    def generateTransmitSignal(self):
        # Calculate waveform duration based on scene geometry
        # Assumes scene center is at
        #(x, y, z) = np.meshgrid(self.xVect, self.yVect, self.zVect)
        #pix3D = np.hstack((np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)), np.reshape(z, (np.size(z), 1))))
        #min_dist = []
        #max_dist = []
        #x = np.ones_like(pix3D)
        #for proj in self.projectors:
        #    proj = proj.detach().cpu().numpy()
        #    dist = np.sum((pix3D - (x*proj))**2, 1)
        #    tofs = 2 * np.sqrt(dist)
        #    min_dist.append(np.min(tofs))
        #    max_dist.append(np.max(tofs))

        #self.minDist = min(min_dist)
        self.minDist = 0.8686305776399169
        #print(self.minDist)
        #self.minDist = 0
        #self.maxDist = max(max_dist)
        #self.tDur = .02
        #self.tDur = (self.maxDist - self.minDist)/self.c
        self.tDur = 0.007467797273989506
        #print(self.tDur)
        #self.nSamples = math.ceil(self.tDur * self.Fs)
        #self.nSamples = self.nSamples + 50 # Hack to get optimization to work
        #self.nSamples = int(math.ceil(self.nSamples/100.0))*100 # round to nearest hundred
        self.nSamples = 600
        #print(self.nSamples)


        sig = np.zeros(self.nSamples)  # Allocate entire receive signal
        #sig[0] = 1
        times = np.linspace(self.tStart, self.tStop - 1 / self.Fs, num=int((self.tStop - self.tStart) * self.Fs))
        LFM = scipy.signal.chirp(times, self.fStart, self.tStop, self.fStop)  # Generate LFM chirp
        window = scipy.signal.tukey(len(LFM), self.winRatio)
        LFM = np.multiply(LFM, window)  # Window chirp to suppress side lobes
        ind1 = 0 # Not supporting staring time other than zero atm
        ind2 = ind1 + len(LFM)
        sig[ind1:ind2] = LFM  # Insert chirp into receive signal

        # Convert transmit signal to tensor
        self.transmitSignal = torch.from_numpy(sig).to(self.dev)
        self.transmitSignal.requires_grad = False

        self.pulse = torchHilbert(self.transmitSignal, self)
        self.Pulse = torch.fft(self.pulse, 1).to(self.dev) # Complex version of the transmitted signal


    def defineProjectorPosGrid(self, **kwargs):
        self.xStart = kwargs.get('xStart', -1)
        self.xStop = kwargs.get('xStop', 1)
        self.yStart = kwargs.get('yStart', -1)
        self.yStop = kwargs.get('yStop', 1)
        self.zStart = kwargs.get('zStart', 0.3)
        self.zStop = kwargs.get('zStop', 0.3)
        self.xStep = kwargs.get('xStep', .1)
        self.yStep = kwargs.get('yStep', .1)
        self.zStep = kwargs.get('zStep', .1)

        self.numXs = int((self.xStop - self.xStart)/self.xStep) + 1
        self.numYs = int((self.yStop - self.yStart)/self.yStep) + 1
        self.numZs = int((self.zStop - self.zStart)/self.zStep) + 1

        self.numProj = int(self.numXs * self.numYs * self.numZs)

        self.xs = np.linspace(self.xStart, self.xStop, self.numXs)
        self.ys = np.linspace(self.yStart, self.yStop, self.numYs)
        self.zs = np.linspace(self.zStart, self.zStop, self.numZs)

        projectors = np.zeros((self.numProj, 2))

        count = 0
        for i in range(0, self.numXs):
            for j in range(0, self.numYs):
                projectors[count, :] = [self.xs[i], self.ys[j]]
                count = count + 1
        self.projectors = torch.from_numpy(projectors).to(self.dev)
        self.projectors.requires_grad = False

    def freeHooks(self, **kwargs):
        N = len(self.hooks)
        for i in range(0, N):
            tmp = self.hooks[i]
            tmp.remove()

    def defineProjectorPos(self, **kwargs):
        self.thetaStart = kwargs.get('thetaStart', 0)
        self.thetaStop = kwargs.get('thetaStop', 359)
        self.thetaStep = kwargs.get('thetaStep', 1)
        self.zStart = kwargs.get('zStart', 0.3)
        self.zStop = kwargs.get('zStop', 0.3)
        self.zStep = kwargs.get('zStep', 0.1)
        self.rStart = kwargs.get('rStart', 1)
        self.rStop = kwargs.get('rStop', 1)
        self.rStep = kwargs.get('rStep', 1)

        # Count number of theta, r, and z values
        self.numThetas = int((self.thetaStop - self.thetaStart) / self.thetaStep) + 1
        self.numRs = int((self.rStop - self.rStart) / self.rStep) + 1
        self.numZs = int((self.zStop - self.zStart) / self.zStep) + 1

        self.numProj = int(self.numThetas * self.numRs * self.numZs)

        # Define array of r, theta, and z values
        self.rs = np.linspace(self.rStart, self.rStop, self.numRs)
        self.thetas = np.linspace(self.thetaStart, self.thetaStop, self.numThetas)
        self.zs = np.linspace(self.zStart, self.zStop, self.numZs)

        # Allocate memory for projector array
        projectors = np.zeros((self.numProj, 3))

        # Pack every projector position into an array
        count = 0
        for i in range(0, self.numThetas):
            for j in range(0, self.numRs):
                for k in range(0, self.numZs):
                    projectors[count, :] = [self.rs[j] * math.cos(np.deg2rad(self.thetas[i])),
                                            self.rs[j] * math.sin(np.deg2rad(self.thetas[i])), self.zs[k]]
                    count = count + 1
        self.projectors = torch.from_numpy(projectors).to(self.dev)
        self.projectors.requires_grad = False


