import torch
from torch.autograd import *
import matplotlib.pyplot as plt
import numpy as np
import random


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


# Vectorize point operations for speed
class DelaySignalBatched(Function):
    """
    @staticmethod
    def forward(ctx, ps, pData, RP):
        ctx.ps = ps.detach().cpu().numpy()
        ctx.numScat = list(ps.shape)[0]
        ctx.projPos = pData.projPos.repeat(ctx.numScat, 1).detach().cpu().numpy()

        ctx.RP = RP
        ctx.tx = RP.transmitSignal.detach().cpu().numpy()

        t = np.sqrt(np.sum((ctx.projPos - ctx.ps[:, :]) ** 2, 1))
        tau = (t * 2) / RP.c
        startIndeces = (tau * RP.Fs).astype(np.int)
        idx = startIndeces[:, None] + range(RP.nSamplesTransmit)
        ctx.indeces = torch.from_numpy(startIndeces)
        ctx.indeces = ctx.indeces.long()
        sig = np.zeros((ctx.numScat, RP.nSamples))
        # https://stackoverflow.com/questions/47516197/select-slices-range-of-columns-for-each-row-in-a-pandas-dataframe
        # Select certain columns per row broadcasting magic
        print(np.arange(len(idx))[:, None].shape)
        print(idx.shape)
        print(ctx.tx.shape)
        sig[np.arange(len(idx))[:, None], idx] = ctx.tx

        sig = torch.from_numpy(sig)
        sig = torch.sum(sig, 0)

        return sig
    """

    @staticmethod
    def forward(ctx, ps, pData, RP):
        ctx.ps = ps
        ctx.numScat = list(ps.shape)[0]
        ctx.projPos = pData.projPos.repeat(ctx.numScat, 1)

        ctx.RP = RP
        ctx.tx = RP.transmitSignal

        t = torch.sqrt(torch.sum((ctx.projPos - ctx.ps)**2, 1))
        tau = (t*2)/RP.c
        startIndeces = (tau*RP.Fs).type(torch.int)

        idx = startIndeces[:, None] + torch.arange(RP.nSamplesTransmit).to(RP.dev).type(torch.long)

        ctx.indeces = startIndeces.type(torch.long)
        sig = torch.zeros([ctx.numScat, RP.nSamples], dtype=torch.float64).to(RP.dev)

        sig[(torch.arange(len(idx))[:, None]).type(torch.long), idx] = ctx.tx

        sig = torch.sum(sig, 0)

        return sig
    # Should weight by positive potential because we want points to stick to ground truth points.
    # Points hould move towards either minimum gradient or some weighted combination of the minimum gradient
    # Need to average if gradient is postive over waveform and then move towards the most negative gradient. Should fix 3D. I think i'm sensitve to noise
    # and overlay from z planes is causing ambiguities.
    @staticmethod
    def backward(ctx, grad_output):
        #projPos = torch.from_numpy(ctx.projPos)
        #ps = torch.from_numpy(ctx.ps)
        projPos = ctx.projPos
        ps = ctx.ps

        # Find all negative gradients in waveform -> this is where points need to move
        neg_grad_indeces = torch.where(grad_output < 0)[0]

        #minVal = torch.min(grad_output, dim=0)[1]
        #minVal = minVal.repeat(len(ctx.indeces))

        rand_index = torch.randint(len(neg_grad_indeces), (ctx.numScat,))

        # Are negative gradients to left or right of point indeces (set to 0 and 1 by int cast)
        sign = (neg_grad_indeces[rand_index] < ctx.indeces).type(torch.int)

        #dist = (ctx.indeces - minVal).to(ctx.RP.dev)

        #b = 1/(1 + (grad_output[ctx.indeces] - grad_output[minVal]).abs())

        #b = (grad_output[ctx.indeces]/grad_output[minVal]).abs()

        #b = (grad_output[ctx.indeces]/grad_output[minVal].abs()).to(ctx.RP.dev)

        # If to the left move tau other direction
        sign[torch.where(sign == 0)] = -1

        # Not sure if this is the most correct way to weight the derivatives. But not the worst
        dTau = sign*((grad_output[neg_grad_indeces[rand_index]]).abs()) # Array of dTau for each point
        #dTau = dist

        #weight = (grad_output[ctx.indeces] > 0).type(torch.int).to(ctx.RP.dev)
        #weight[torch.where(weight == 0)] = -1

        # Derivative of euclidean distance
        const = dTau * -1 * torch.pow(torch.sum((projPos - ps) ** 2), -.5)
        ps_grad = torch.transpose(const.repeat(3, 1), 0, 1) * ((projPos - ps))

        return ps_grad.to(ctx.RP.dev), None, None


    """
    @staticmethod
    def backward(ctx, grad_output):
        ps_grad = np.zeros_like(ctx.ps)
        #print(ctx.projPos)
        gradients = grad_output.detach().cpu().numpy()

        #plt.clf()
        #plt.stem(grad_output.detach().cpu().numpy(), use_line_collection=True)
        #plt.savefig('test_pics/look.png')
        #result = np.where(gradients == np.amin(gradients))
        #ind = torch.tensor(result[0], dtype=torch.long)
        #print(ind)
        #plt.clf()
        #plt.stem(gradients, use_line_collection=True)
        #plt.savefig('test_pics/look.png')
        indeces = torch.where(grad_output < 0)
        for i in range(0, ctx.numScat):
            if torch.sum(grad_output[ctx.indeces[i]:ctx.indeces[i] + ctx.RP.nSamplesTransmit]) > 0:
                print("P")
                #if ind < ctx.indeces[i]:
                #    dTau = grad_output[ctx.indeces[i]].abs()
                #elif ind > ctx.indeces[i]:
                #    dTau = -grad_output[ctx.indeces[i]].abs()
                #else:
                #    dTau = 0

                ind = random.choice(indeces[0])

                if ind < ctx.indeces[i]:
                    dTau = grad_output[ctx.indeces[i]].abs()
                elif ind > ctx.indeces[i]:
                    dTau = -grad_output[ctx.indeces[i]].abs()
                else:
                    dTau = 0
                #left_indices = torch.where(grad_output[0:ctx.indeces[i]] < 0)
                #right_indices = torch.where(grad_output[ctx.indeces[i]:] < 0)[0] + ctx.indeces[i]

                #leftSum = torch.sum(torch.abs(grad_output[left_indices]))
                #rightSum = torch.sum(torch.abs(grad_output[right_indices]))

                #dTau = (leftSum-rightSum) * grad_output[ctx.indeces[i]].abs()
            else:
                print("N")
                dTau = 0
                #flip = 1
            #elif torch.sum(grad_output[ctx.indeces[i]:ctx.indeces[i] + ctx.RP.nSamplesTransmit]) < 0:
            #    left_indices = torch.where(grad_output[0:ctx.indeces[i]] < 0)
            #    right_indices = torch.where(grad_output[ctx.indeces[i]:] < 0)[0] + ctx.indeces[i]

            #    leftSum = torch.sum(torch.abs(grad_output[left_indices]))
            #    rightSum = torch.sum(torch.abs(grad_output[right_indices]))

            #    dTau = (leftSum - rightSum) * grad_output[ctx.indeces[i]].abs()

            #    flip = 1
            #else:
            #    dTau = 0
            #    flip = 0
            #a = torch.where(grad_output[0:ctx.indeces[i]] < 0)
            #ind = 1
            #dTau = -torch.sum(grad_output)

            #index = ctx.indeces[i].type(torch.DoubleTensor)
            #indeces = torch.where(grad_output[:] < 0)
            #print(grad_output.unsqueeze(0).shape)
            #print(gradients.shape)
            #min, ind = np.min(gradients)

            #min, ind = torch.min(grad_output.unsqueeze(0))
            #dTau = torch.sum(grad_output[ctx.indeces[i]:ctx.indeces[i]+ctx.RP.nSamplesTransmit])




            #left_indeces = li[0].type(torch.DoubleTensor)
            #right_indeces = ri[0].type(torch.DoubleTensor) + index.type(torch.DoubleTensor)
            #print(right_indeces)
            #if ind < ctx.indeces[i]:
            #    dTau = 1
            #elif ind > ctx.indeces[i]:
            #    dTau = -1
            #else:
            #    dTau = 0

            #if right_indeces.numel() == 0:
            #    right_indeces = torch.tensor(ctx.RP.nSamples*1.0)
            #if left_indeces.numel() == 0:
            #    left_indeces = torch.tensor(0.0)

            #leftDist = torch.abs(index - torch.mean(left_indeces))
            #rightDist = torch.abs(torch.mean(right_indeces) - index)

            #print(leftDist, rightDist)

            #if leftDist < rightDist:
            #    dTau = flip * torch.sum(torch.abs(grad_output[li]))
            #if rightDist < leftDist:
            #    dTau = -1*flip * torch.sum(torch.abs(grad_output[index + ri]))
            #if rightDist == leftDist:
            #    dTau = flip * torch.sum(torch.abs(grad_output[li]))

            #if torch.is_tensor(dTau):
            #    dTau = dTau.detach().cpu().numpy()
            #print(dTau)
            const = dTau * -1 * np.power(np.sum((ctx.projPos[0, :] - ctx.ps[i, :]) ** 2), -.5)
            ps_grad[i, :] = const * (ctx.projPos[0, :] - ctx.ps[i, :])
        ps_grad = torch.from_numpy(ps_grad)
        #print(ps_grad)
        #print(ps_grad[0, :])

        return ps_grad.to(ctx.RP.dev), None, None
"""


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
            tau = (t * 2) / RP.c
            sig = torch.zeros(RP.nSamples)
            timeIndex = (tau * RP.Fs).long()
            ctx.timeIndeces.append(timeIndex)

            sig[timeIndex:timeIndex + RP.nSamplesTransmit] = RP.transmitSignal
            final_sig += sig
        # plt.clf()
        # plt.stem(final_sig.detach().cpu().numpy(), use_line_collection=True)
        # plt.show()
        return final_sig

    """
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output.shape)
        # Normalize gradient output so that each projector pulls on the point equally (helps convergence)
        ps_grad = torch.zeros_like(ctx.ps)
        #grad_output = 2*((grad_output - grad_output.min().detach())/(grad_output.max().detach() - grad_output.min().detach())) - 1
        for i in range(0, ctx.numScat):
            if grad_output[ctx.timeIndeces[i]] > 0:
                #print('positive')
                left_indices = torch.where(grad_output[0:ctx.timeIndeces[i]] < 0)
                right_indices = torch.where(grad_output[ctx.timeIndeces[i]:] < 0)[0] + ctx.timeIndeces[i]

                leftSum = torch.sum(torch.abs(grad_output[left_indices]))
                rightSum = torch.sum(torch.abs(grad_output[right_indices]))

                dTau = (leftSum - rightSum)*grad_output[ctx.timeIndeces[i]].abs()#\
                       #*(grad_output[ctx.timeIndeces[i]].abs())
            elif grad_output[ctx.timeIndeces[i]] < 0:
                #print('negative')
                #print(ctx.timeIndeces[i])
                #plt.clf()
                #plt.stem(grad_output.detach().cpu().numpy(), use_line_collection=True)
                #plt.savefig('test_pics/look.png')
                left_indices = torch.where(grad_output[0:ctx.timeIndeces[i]] > 0)
                right_indices = torch.where(grad_output[ctx.timeIndeces[i]:] > 0)[0] + ctx.timeIndeces[i]

                leftSum = torch.sum(torch.abs(grad_output[left_indices]))
                rightSum = torch.sum(torch.abs(grad_output[right_indices]))

                dTau = (rightSum - leftSum)*grad_output[ctx.timeIndeces[i]].abs()#\
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
"""
