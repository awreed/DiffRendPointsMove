import torch

class VectorDistribution:
    def __init__(self, T):
        self.dist = T/torch.norm(T, p=1)
        self.mean = self.calcMean()

    def calcMean(self):
        # index 1 contains the other dim if complex array
        N = list(self.dist.shape)[0]
        indexVec = torch.linspace(1, N, N)
        return torch.sum(indexVec * self.dist)
