import torch

class Complex:
    def __init__(self, **kwargs):
        self.real = kwargs.get('real', None)
        self.imag = kwargs.get('imag', None)

    def vector(self):
        return torch.stack((self.real, self.imag), 1)

    def abs(self):
        return torch.sqrt((self.real**2 + self.imag**2))

    def conj(self):
        imag = -1 * self.imag
        return torch.stack((self.real, imag), 1)

    @staticmethod
    def multiply(c1, c2):
        return torch.stack((c1.real*c2.real - c1.imag*c2.imag, c1.imag*c2.real+c1.real*c2.imag), 1)

