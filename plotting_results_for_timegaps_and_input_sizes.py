import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.distributions import Categorical 
import random
from statistics import mean
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import math
plt.rcParams['font.size'] = 18
plt.rcParams["font.family"] = "Times New Roman"

a = [1,2,3,4,5,6,7,8,9]
hidden24_vanilla = [31.52, 53.22, 38.10, 37.04, 67.64, 38.18, 61.40, 75.66, 55.56]
hidden24_neuronal_STPN = [47.64, 52.30, 60.72, 51.28, 59.07, 72.48, 54.82, 72.38, 80.16]
hidden24_synaptic_STPN = [51.31, 54.68, 67.35, 54.06, 62.11, 73.12, 59.44, 74.52, 80.35]
hidden24_neuronal_STPN_dynamicz = [50.61, 51.42, 60.67, 46.06, 59.48, 74.71, 53.73, 73.59, 80.79]
hidden24_synaptic_STPN_dynamicz = [50.39, 51.78, 69.48, 47.82, 58.77, 74.35, 56.28, 73.58, 82.00]
hidden24_bio_GRU = [56.21, 57.30, 72.94, 50.42, 64.54, 82.13, 61.56, 74.79, 84.81]
hidden24_simple_GRU = [21.50, 74.28, 85.23, 73.46, 71.77, 82.64, 74.28, 84.10, 92.58]
hidden24_GRU = [75.34, 82.30, 86.82, 76.81, 72.85, 90.57, 82.20, 88.49, 93.26]

hidden48_vanilla = [40.47, 45.38, 39.67, 38.66, 36.26, 55.24, 45.69, 75.48, 66.43]
hidden48_neuronal_STPN = [50.92, 55.08, 69.37, 54.98, 61.06, 79.86, 60.98, 75.68, 83.99]
hidden48_synaptic_STPN = [39.02, 57.07, 75.73, 54.60, 63.54, 80.04, 61.83, 78.21, 86.37]
hidden48_neuronal_STPN_dynamicz = [55.22, 55.0, 68.72, 56.54, 62.54, 78.15, 62.17, 77.54, 86.79]
#hidden48_synaptic_STPN_dynamicz = 
hidden48_bio_GRU = [60.15, 64.32, 79.66, 60.23, 69.53, 86.15, 67.03, 77.93, 88.46]
hidden48_simple_GRU = [83.83, 65.31, 89.68, 83.48, 85.72, 94.07, 85.78, 73.16, 85.91]

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
my_xticks = ['4_1','4_4','4_28','8_1', '8_4', '8_28', '16_1', '16_4', '16_28']
plt.xticks(a, my_xticks)
plt.plot(a,hidden24_vanilla)#, 'cyan')
plt.plot(a,hidden24_neuronal_STPN)#, 'pink')
plt.plot(a,hidden24_synaptic_STPN)#, 'red')
plt.plot(a,hidden24_bio_GRU)#, 'green')
plt.plot(a,hidden24_simple_GRU)#, 'green')
plt.plot(a,hidden24_GRU)#, 'black')
plt.plot(a, hidden24_neuronal_STPN_dynamicz)
plt.plot(a, hidden24_synaptic_STPN_dynamicz)
#plt.plot(a,hidden48_vanilla, 'cyan')
#plt.plot(a,hidden48_neuronal_STPN, 'pink')
#plt.plot(a,hidden48_synaptic_STPN, 'red')
#plt.plot(a,hidden48_bio_GRU, 'green')
plt.legend(['Vanilla RNN', 'Neuronal STPN', 'Synaptic STPN', 'Bio GRU', 'Simple GRU', 'GRU', 'Bio Neuronal STPN', 'Bio Synaptic STPN'], prop={'size': 20})
plt.ylabel('Test accuracies %')
plt.show()

my_xticks = ['4_1','4_4','4_28','8_1', '8_4', '8_28', '16_1', '16_4', '16_28']
plt.xticks(a, my_xticks)
plt.plot(a,hidden48_vanilla)#, 'cyan')
plt.plot(a,hidden48_neuronal_STPN)#, 'pink')
plt.plot(a,hidden48_synaptic_STPN)#, 'red')
plt.plot(a,hidden48_bio_GRU)#, 'green')
plt.plot(a,hidden48_simple_GRU)#, 'green')
plt.plot(a,hidden24_GRU)#, 'black')
plt.legend(['Vanilla RNN', 'Neuronal STPN', 'Synaptic STPN', 'Bio GRU', 'Simple GRU', 'GRU (hidden size 24)'], prop={'size': 20})
plt.ylabel('Test accuracies %')
plt.show()