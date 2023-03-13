import torch
import math
import matplotlib.pyplot as plt
import numpy as np 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
}
loaders

from torch import nn
import torch.nn.functional as F      

sequence_length = 196
input_size = 4
hidden_size = 24
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01


class STPCell(nn.Module):
    def __init__(self, input_size, hidden_size, complexity, e_h, alpha):
        super(STPCell, self).__init__()
        self.input_size = input_size        
        self.hidden_size = hidden_size
        self.complexity = complexity 
        sigmoid = nn.Sigmoid() 
        self.ones = torch.ones(self.hidden_size, self.hidden_size)
        self.batch_size = batch_size 
        self.forprintingX = []
        self.forprintingU = []
        self.forprintingh = []

        if self.complexity == "rich":
            # System variables 
            self.e_h = e_h

            # Short term Plasticity variables 
            self.delta_t = 1
            self.alpha = alpha
            self.e_ux = self.alpha * self.e_h
            self.z_min = 0.001
            self.z_max = 0.1

            # Short term Depression parameters  
            self.c_x = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))

            # Short term Facilitation parameters
            self.c_u = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
            self.c_U = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
            
            # System parameters            
            self.c_h = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            self.w = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
            self.p = torch.nn.Parameter(torch.rand(self.hidden_size, self.input_size))  
            self.b = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            
            # State initialisations
            self.h_t = torch.zeros(1, self.hidden_size, dtype=torch.float32)
            self.X = torch.ones(self.hidden_size, self.hidden_size, dtype=torch.float32)     
            self.U = torch.full((self.hidden_size, self.hidden_size), 0.9, dtype=torch.float32)         
            self.Ucap = 0.9 * sigmoid(self.c_U)
            self.Ucapclone = self.Ucap.clone().detach()
        if self.complexity == "poor":
            # System variables 
            self.e_h = e_h

            # Short term Plasticity variables 
            self.delta_t = 1
            self.alpha = alpha
            self.e_ux = self.alpha * self.e_h

            # Short term Depression parameters  
            self.c_x = torch.nn.Parameter(torch.rand(self.hidden_size, 1))

            # Short term Facilitation parameters
            self.c_u = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            self.c_U = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            
            # System parameters
            self.c_h = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            self.w = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
            self.p = torch.nn.Parameter(torch.rand(self.hidden_size, self.input_size))
            self.b = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            
            # State initialisations
            self.h_t = torch.zeros(1, self.hidden_size, dtype=torch.float32)
            self.X = torch.ones(self.hidden_size, self.hidden_size, dtype=torch.float32)
            self.U = torch.full((x.size(0), self.hidden_size, self.hidden_size), 0.9, dtype=torch.float32)
            self.Ucap = 0.9 * sigmoid(self.c_U)
            self.Ucapclone = self.Ucap.clone().detach()
        for name, param in self.named_parameters():
            print(name, param.size(), param)
            nn.init.uniform_(param, a=-(1/math.sqrt(hidden_size)), b=(1/math.sqrt(hidden_size))) 

    def forward(self, x):                    
        if self.complexity == "rich":
            if self.h_t.dim() == 3:
                self.h_t = self.h_t[0]
            self.h_t = torch.transpose(self.h_t, 0, 1)
            x = torch.transpose(x, 0, 1)
            sigmoid = nn.Sigmoid()
            
            # graph plotting 
            '''self.forprintingX.append(self.X[20,11,24].item())
            self.forprintingU.append(self.U[20,11,24].item())
            self.forprintingh.append(self.h_t[11, 20].item())
            if len(self.forprintingX) % (196*5) == 0:
                self.forprintingX = []
                self.forprintingU = []
                self.forprintingh = []'''   

            # Short term Depression 
            self.z_x = self.z_min + (self.z_max - self.z_min) * sigmoid(self.c_x)
            #print("z_x", self.z_x.size())
            #print("self.X", self.X.size())
            #print("self.ones", self.ones.size())
            #print("h_t", self.h_t.size())
            #a = self.delta_t * self.U * torch.einsum("ijk, ji  -> ijk", self.X, self.h_t)
            #print("a", a)
            #print("a size", a.size())
            self.X = self.z_x + torch.mul((1 - self.z_x), self.X) - self.delta_t * self.U * torch.einsum("ijk, ji  -> ijk", self.X, self.h_t)

            # Short term Facilitation 
            self.z_u = self.z_min + (self.z_max - self.z_min) * sigmoid(self.c_u)    
            self.Ucap = 0.9 * sigmoid(self.c_U)
            self.U = self.Ucap * self.z_u + torch.mul((1 - self.z_u), self.U) + self.delta_t * self.Ucap * torch.einsum("ijk, ji  -> ijk", (1 - self.U), self.h_t)
            self.Ucapclone = self.Ucap.clone().detach() 
            self.U = torch.clamp(self.U, min=self.Ucapclone.repeat(self.batch_size, 1, 1).to(device), max=torch.ones_like(self.Ucapclone.repeat(self.batch_size, 1, 1).to(device)))

            # System Equations 
            self.z_h = self.e_h * sigmoid(self.c_h) 
            #   a = self.w * self.U * self.X
            #print("size of a", a.size())
            #print("size of h_t", self.h_t.size())
            #print("size of a * h_t", torch.matmul(a, self.h_t).size())
            #print("size of x", x.size())
            x = torch.transpose(x, 0, 1)
            self.h_t = torch.mul((1 - self.z_h), self.h_t) + self.z_h * sigmoid(torch.einsum("ijk, ki  -> ji", (self.w * self.U * self.X), self.h_t) + torch.matmul(self.p, x) + self.b)
            #self.h_t = torch.matmul(self.w, self.h_t) + torch.matmul(self.p, x) + self.b
            self.h_t = torch.transpose(self.h_t, 0, 1)
            return self.h_t   

        if self.complexity == "poor":
            if self.h_t.dim() == 3:
                self.h_t = self.h_t[0]
            self.h_t = torch.transpose(self.h_t, 0, 1)
            x = torch.transpose(x, 0, 1)
            sigmoid = nn.Sigmoid()
            
            # Short term Depression 
            self.z_x = self.z_min + (self.z_max - self.z_min) * sigmoid(self.c_x)
            #print("z_x", self.z_x.size())
            #print("self.X", self.X.size())
            #print("self.ones", self.ones.size())
            #print("h_t", self.h_t.size())
            #a = self.delta_t * self.U * self.X * self.h_t
            #print("a", a)
            #print("a size", a.size())
        
            self.X = self.z_x + torch.mul((1 - self.z_x), self.X) - self.delta_t * self.U * self.X * self.h_t

            # Short term Facilitation 
            self.z_u = self.z_min + (self.z_max - self.z_min) * sigmoid(self.c_u)    
            self.Ucap = 0.9 * sigmoid(self.c_U)
            self.U = self.Ucap * self.z_u + torch.mul((1 - self.z_u), self.U) + self.delta_t * self.Ucap * (1 - self.U) * self.h_t
            self.Ucapclone = self.Ucap.clone().detach()
            self.U = torch.clamp(self.U, min=self.Ucapclone.repeat(self.batch_size, 1, 1).to(device), max=torch.ones_like(self.Ucapclone.repeat(self.batch_size, 1, 1).to(device)))
            # graph plotting 
            '''self.forprintingX.append(self.X[20,5].item())
            self.forprintingU.append(self.U[20,5].item())
            if len(self.forprintingX) % 140 == 0:
                self.forprintingX = []
                self.forprintingU = []'''

            # System Equations 
            # self.z_h = self.e_h * sigmoid(self.c_h) 
            #a = self.w * self.U * self.X
            #print("size of a", a.size())
            #print("size of h_t", self.h_t.size())
            #print("size of a * h_t", torch.matmul(a, self.h_t).size())
            #print("size of x", x.size())
            x = torch.transpose(x, 0, 1)
            self.h_t = torch.mul((1 - self.c_h), self.h_t) + self.c_h * sigmoid(torch.matmul(self.w, (self.U * self.X * self.h_t)) + torch.matmul(self.p, x) + self.b)
            #self.h_t = torch.matmul(self.w, self.h_t) + torch.matmul(self.p, x) + self.b
            self.h_t = torch.transpose(self.h_t, 0, 1)
            return self.h_t

class STP(nn.Module):
    def __init__(self, input_size, hidden_size, complexity, e_h, alpha): 
        super(STP, self).__init__()
        self.stpcell = STPCell(input_size, hidden_size, complexity, e_h, alpha).to(device)

    def forward(self, x):
        for n in range(x.size(1)):
            x_slice = torch.transpose(x[:,n,:], 0, 1)
            self.stpcell(x_slice)
        return self.stpcell.h_t                                   
            
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = STP(input_size, hidden_size, "rich", 0.9, 0.1).to(device)
        self.fc = nn.Linear(hidden_size, num_classes).to(device)
        self.update_number = 0
        pass

    def forward(self, x):
        # Set initial hidden and cell states 
        if self.lstm.stpcell.complexity == "rich":
            self.lstm.stpcell.h_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            self.lstm.stpcell.X = torch.ones(x.size(0), self.hidden_size, self.hidden_size, dtype=torch.float32).to(device)
            #self.lstm.stpcell.U = torch.full((x.size(0), self.hidden_size, self.hidden_size), 0.9, dtype=torch.float32).to(device)
            self.lstm.stpcell.U = (self.lstm.stpcell.Ucapclone.repeat(self.lstm.stpcell.batch_size, 1, 1)).to(device)
        if self.lstm.stpcell.complexity == "poor":
            self.lstm.stpcell.h_t = torch.full((self.num_layers, x.size(0), self.hidden_size), 0.9).to(device) 
            self.lstm.stpcell.X = torch.ones(self.hidden_size, x.size(0), dtype=torch.float32).to(device)
            #self.lstm.stpcell.U = torch.full((self.hidden_size, x.size(0)), 0.9, dtype=torch.float32).to(device)
            self.lstm.stpcell.U = (self.lstm.stpcell.Ucapclone.repeat(self.lstm.stpcell.batch_size, 1, 1)).to(device)
            #torch.full((2, 3), 3.141592)
        '''self.update_number += 1 
        if self.update_number % 50 == 0: 
            plt.plot(self.lstm.stpcell.forprintingX)
            plt.plot(self.lstm.stpcell.forprintingU)
            plt.plot(self.lstm.stpcell.forprintingh)
            plt.legend(["X","U","h_t"])
            plt.show()'''
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        return out
        
        pass                                    
pass
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)
loss_func = nn.CrossEntropyLoss()

from torch import optim
optimizer = optim.Adam(model.parameters(), lr = 0.01)   

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
    spiraltensor = spiraltensor.reshape(28, 28).to(device)
    return spiraltensor

def baseindexing(time_gap, input_size, stride, a):
    a = a.flatten()
    a = a.cpu()
    a = a.numpy()

    baseinds = np.arange(0, time_gap*input_size, time_gap)

    #zero padding 
    a = np.pad(a, (baseinds[-1],0), 'constant')

    new_sequence = []

    for t in range(784):
        new_sequence.append(a[(t+baseinds).tolist()])        
    '''print("baseinds", baseinds)
    print("2baseinds", 2+baseinds)
    print((2+baseinds).tolist())
    print(a[(2+baseinds).tolist()])'''
    
    #new_sequence = [item for sublist in new_sequence for item in sublist] 

    new_sequence = torch.tensor(new_sequence, dtype=torch.float).to(device)
    #print("size of new sequence tensor", new_sequence.size())
    return new_sequence

def train(num_epochs, model, loaders): 
        
    # Train the model
    total_step = len(loaders['train'])
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            p = torch.rand(batch_size, 1, 784, input_size)
            for n in range(images.size(dim=0)):         # Use this loop to spiralise the image 
                for image in images[n,0:1,:,:]: 
                    spiralimage = spiraliser(28, 28, image)        
                    indexedimage = baseindexing(3, input_size, 1, spiralimage)  
                    p[n,0,:,:] = indexedimage
                    #print(images[2,0:1,0:28,0:28])
                    #print(p.size())
                    #print(image.size())
            images = p.clone()     
            images = images.reshape(-1, 784, input_size).to(device)
            #images = images.reshape(-1, input_size*28*28, input_size).to(device)
            labels = labels.to(device)
            # Forward pass    
            outputs = model(images)
            loss = loss_func(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))  
                pass
        
        pass
    pass
train(num_epochs, model, loaders)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loaders['test']:

        #spiral 
        p = torch.rand(batch_size, 1, 784, input_size)
        for n in range(images.size(dim=0)):         # Use this loop to spiralise the image 
            for image in images[n,0:1,:,:]: 
                spiralimage = spiraliser(28, 28, image)        
                indexedimage = baseindexing(3, input_size, 1, spiralimage)  
                p[n,0,:,:] = indexedimage
        images = p.clone()

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

sample = next(iter(loaders['test']))
imgs, lbls = sample 

test_output = model(imgs[:10].view(-1, 28, 28))
predicted = torch.max(test_output, 1)[1].data.numpy().squeeze()
labels = lbls[:10].numpy()
print(f"Predicted number: {predicted}")
print(f"Actual number: {labels}")

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 2
imgs_print = imgs[:10]  
for i in range(1, cols * rows + 1):
    figure.add_subplot(rows, cols, i)
    plt.title(predicted[i-1])
    plt.axis("off")
    plt.imshow(imgs_print[i-1].squeeze(), cmap="gray")
plt.show()