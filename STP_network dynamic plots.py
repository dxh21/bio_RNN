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

# As PIL.Image
dataset = datasets.MNIST(root='data')
x, _ = dataset[1111]
x.show() # x is a PIL.Image here

# As torch.Tensor
dataset = datasets.MNIST(
    root='data',
    train = False,
    transform=ToTensor()
)

x, _ = dataset[1] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
plt.show()