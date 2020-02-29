import numpy as np
from scipy.signal import correlate
from scipy.signal import hilbert
import torch
from utils import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from Complex import *

class ProjData:
    def __init__(self, *args, **kwargs):
        self.projPos = kwargs.get('projPos', [0, 0, 0])
        self.Fs = kwargs.get('Fs', 100000)
        self.tDur = kwargs.get('tDur', .02)
        #self.wfm = torch.zeros((int(self.Fs*self.tDur)), requires_grad=True)
        self.wfm = None
        self.wfmRC = None
        self.normWfmRC = None
        self.t = None
        self.tau = None
        self.wfms = []

    def RC(self, transmitSignal):
        # Replica-correlate using the fft
        nSamples = self.Fs*self.tDur
        print(nSamples)
        pulse = transmitSignal
        Pulse = np.fft.fft(hilbert(pulse), int(nSamples))
        Data = np.fft.fft(hilbert(self.wfm), int(nSamples))
        yRC = np.fft.ifft(np.multiply(Data, np.conj(Pulse)))


    def RCTorch(self, RP):
        #nSamples = self.Fs * self.tDur
        Pulse = RP.Pulse


        # Forward fourier transform of received waveform
        DataHil = torchHilbert(self.wfm)
        Data = torch.fft(DataHil, 1)

        # Definition of cross-correlation
        yRC = torch.ifft(compMul(Data, compConj(Pulse)), 1)
        self.wfmRC = Complex(real=yRC[:, 0].cuda(), imag=yRC[:, 1].cuda())
        self.normWfmRC = self.wfmRC.abs()/torch.norm(self.wfmRC.abs(), p=1.0)


