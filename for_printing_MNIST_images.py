import torch
import math
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd
from IPython.display import display
from mpl_toolkits import mplot3d
from torchvision import datasets
from torchvision.transforms import ToTensor

# As torch.Tensor
dataset = datasets.MNIST(
    root='data',
    train = False,
    transform=ToTensor()
)

x, _ = dataset[110] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
plt.show()

def spiraliser(m, n, a):
    a = a.cpu()
    a = a.numpy()
    k = 0
    l = 0
    spiral = []
    ''' k - starting row index
        m - ending row index
        l - starting column index
        n - ending column index
        i - iterator '''
  
    while (k < m and l < n):
  
        # Print the first row from
        # the remaining rows
        for i in range(l, n):
            spiral.append(a[k][i])
  
        k += 1
  
        # Print the last column from
        # the remaining columns
        for i in range(k, m):
            spiral.append(a[i][n - 1])
  
        n -= 1
  
        # Print the last row from
        # the remaining rows
        if (k < m):
  
            for i in range(n - 1, (l - 1), -1):
                spiral.append(a[m - 1][i])
  
            m -= 1
  
        # Print the first column from
        # the remaining columns
        if (l < n):
            for i in range(m - 1, k - 1, -1):
                spiral.append(a[i][l])
  
            l += 1
        
    spiraltensor = torch.tensor(spiral, dtype=torch.float)
    spiraltensor = spiraltensor.reshape(28, 28)
    return spiraltensor

x_spiral = spiraliser(28, 28, x[0])
plt.imshow(x_spiral.numpy(), cmap='gray')
plt.show()