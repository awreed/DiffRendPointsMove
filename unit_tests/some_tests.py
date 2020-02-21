"""

# print(tauEST)
# print(np.sum(torchTimeDelay(RP.transmitSignal, RP.Fs, tau).detach().numpy() - timeDelay(RP.transmitSignal.detach().numpy(), RP.Fs, tau))**2)


# plt.stem(pData.wfm.detach().numpy())
# plt.show()

# plt.stem(timeDelay(RP.transmitSignal.detach().numpy(), RP.Fs, tau))
# plt.show()


# t = np.sqrt(np.sum(np.power(pData.projPos.detach().numpy() - ps[0, :], 2)))
# tau = (t * 2)/RP.c
# pData.wfm = pData.wfm + timeDelay(RP.transmitSignal.detach().numpy(), RP.Fs, tau)

# plt.stem(pData.wfm)
# plt.show()

# pData.RC(RP.transmitSignal.detach().numpy())

# plt.stem(pData.wfmRC.real)
# plt.show()


    RP1 = RenderParameters()

    RP1.generateTransmitSignal()

    print(np.shape(RP1.transmitSignal))
    hGT = torchHilbert(RP1.transmitSignal)

    sigEst = torch.zeros(RP1.transmitSignal.shape, requires_grad=True)

    optimizer = torch.optim.Adam([sigEst], lr=.01)

    thresh = .01
    loss = 50

    while abs(loss) > thresh:
        optimizer.zero_grad()
        hEst = torchHilbert(sigEst)
        loss = (hEst - hGT).pow(2).sum()
        loss.backward(retain_graph=True)
        print(loss)
        optimizer.step()

    plt.plot(RP1.transmitSignal.detach().numpy(), color='blue')
    plt.plot(sigEst.detach().numpy(), color='red')
    plt.show()

# RP1.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=1, rStop=1, zStart=.3, zStop=.3)

# pData1 = ProjData(projPos=RP1.projectors[0, :], Fs=RP1.Fs, tdur=RP1.tDur)

# x1 = torchHilbert(RP1.transmitSignal)
# y1 = torchHilbert(sigEst)


#(x, y) = genCos(10)
#(freq, Y) = genCosFFT(y)

#Y_GT = torch.empty(100, 2)
#Y_GT[:, 0] = torch.from_numpy(Y.real)
#Y_GT[:, 1] = torch.from_numpy(Y.imag)

#thresh = .01
#loss = 50
#learning_rate = .01
#y_EST = torch.zeros(100, requires_grad=True)
#optimizer = torch.optim.Adam([y_EST], lr=.01)

#while abs(loss) > thresh:
#    optimizer.zero_grad()
#    Y_EST_FFT = torch.rfft(y_EST, 1, onesided=False)
#    loss = (Y_GT - Y_EST_FFT).pow(2).sum()

    loss.backward()
    print(loss)
    optimizer.step()

y_Fin = y_EST.detach().numpy()
plt.stem(x, y_Fin)
plt.show()

Y_Fin = np.fft.fft(y_Fin)
freq = np.fft.fftfreq(Y_Fin.shape[-1])
freq = freq * fs
plt.stem(freq, Y_Fin.real)
plt.show()

# print(Y_GT.shape)

# y_torch = torch.from_numpy(y)
# Y_torch = torch.rfft(y_torch, 1, onesided=False)

# Y_torch = Y_torch.numpy()
# Y_torch = Y_torch[..., 0] + 1j * Y_torch[..., 1]

# print(type(Y_torch))
# print(np.shape(Y_torch))

# plt.stem(freq, Y_torch)
# plt.show()

# plt.stem(x, y, use_line_collection=True)
# plt.show()

#
# plt.stem(freq, Y.real, use_line_collection=True)
# plt.show()

# x = torch.zeros(100, requires_grad=True)
# Y_GT = torch.from_numpy(Y)
# print(Y_GT.size())
"""