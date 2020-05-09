import torch
from torch.autograd import *
import matplotlib.pyplot as plt

class DelaySignalBatched(Function):
    @staticmethod
    def forward(ctx, ps, pData, RP):
        ctx.ps = ps
        ctx.numScat = list(ps.shape)[0]
        ctx.pData = pData
        ctx.timeIndeces = []
        ctx.RP = RP

        t = torch.sqrt(torch.sum((pData.projPos.repeat(ctx.numScat, 1) - ps[:, :])**2, 1))

        tau = (t*2)/RP.c

        timeIndeces = (tau*RP.Fs).long()

        sig = torch.zeros((ctx.numScat, RP.nSamples))
        sig[:, timeIndeces:timeIndeces+RP.nSamplesTransmit] = RP.transmitSignal
        sig = torch.sum(sig, 0)
        return sig
        print(sig.shape)
    def backward(ctx, grad_output):
        print("Shitty titty")





class DelaySignal(Function):
    @staticmethod
    def forward(ctx, ps, pData, RP):
        ctx.ps = ps
        ctx.numScat = list(ps.shape)[0]
        ctx.pData = pData
        ctx.timeIndeces = []
        ctx.RP = RP
        final_sig = torch.zeros(RP.nSamples)

        for i in range(0, ctx.numScat):
            t = torch.sqrt(torch.sum((pData.projPos - ctx.ps[i, :]) ** 2))
            tau = (t*2)/RP.c
            sig = torch.zeros(RP.nSamples)
            timeIndex = (tau * RP.Fs).long()
            ctx.timeIndeces.append(timeIndex)

            sig[timeIndex:timeIndex + RP.nSamplesTransmit] = RP.transmitSignal
            final_sig += sig
        return final_sig

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output.shape)
        # Normalize gradient output so that each projector pulls on the point equally (helps convergence)
        ps_grad = torch.zeros_like(ctx.ps)
        #grad_output = 2*((grad_output - grad_output.min().detach())/(grad_output.max().detach() - grad_output.min().detach())) - 1
        for i in range(0, ctx.numScat):
            if grad_output[ctx.timeIndeces[i]] > 0:
                #print('here1')
                left_indices = torch.where(grad_output[0:ctx.timeIndeces[i]] < 0)
                right_indices = torch.where(grad_output[ctx.timeIndeces[i]:] < 0)[0] + ctx.timeIndeces[i]

                leftSum = torch.sum(torch.abs(grad_output[left_indices]))
                rightSum = torch.sum(torch.abs(grad_output[right_indices]))

                dTau = (leftSum - rightSum)#\
                       #*(grad_output[ctx.timeIndeces[i]].abs())
            else:
                #print('here')
                left_indices = torch.where(grad_output[0:ctx.timeIndeces[i]] > 0)
                right_indices = torch.where(grad_output[ctx.timeIndeces[i]:] > 0)[0] + ctx.timeIndeces[i]

                leftSum = torch.sum(torch.abs(grad_output[left_indices]))
                rightSum = torch.sum(torch.abs(grad_output[right_indices]))

                dTau = (rightSum - leftSum)#\
                       #*(grad_output[ctx.timeIndeces[i]].abs())



            #print(grad_output[right_indices])
            #print(leftSum)
            #print(rightSum)
            #print("New#############################")
            #print(left_indices)
            #print(right_indices)
            #print(leftSum)
            #print(rightSum)

            #print(dTau)

            #print("dTau is " + str(dTau))
            const = dTau*-1*torch.pow(torch.sum((ctx.pData.projPos - ctx.ps[i, :])**2), -.5)
            #print(const)
            ps_grad[i, :] = const*(ctx.pData.projPos - ctx.ps[i, :])
            #print(ps_grad[i, :])

        #plt.clf()
        #plt.stem(grad_output.detach().cpu().numpy(), use_line_collection=True)
        #plt.show()
        #print("here")
        return ps_grad.to(ctx.RP.dev), None, None
