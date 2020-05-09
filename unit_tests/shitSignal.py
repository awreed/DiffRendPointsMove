import torch
import numpy as np
import matplotlib.pyplot as plt

nSamples = 2000

sig = np.zeros(2000)

sig[0] = 1

plt.stem(sig, use_line_collection=True)
plt.show()

