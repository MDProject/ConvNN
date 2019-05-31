import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os.path as IO
import numpy as np
import matplotlib.pyplot as plt

FC1 = []
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.conv3 = nn.Conv2d(10, 20, 5)
        self.conv4 = nn.Conv2d(20, 10, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #conv1.append(x)
        x = F.relu(self.conv2(x))
        #conv2.append(x)
        x = F.relu(self.conv3(x))
        #conv3.append(x)
        x = F.relu(self.conv4(x))
        #conv4.append(x) 
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        FC1.append(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

batchSize = 50
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# print(IO.exists('../MNIST_DATA/train-labels-idx1-ubyte'))
train_set = dset.MNIST('../MNIST_DATA/', train=True, transform=trans, download=True)
test_set = dset.MNIST('../MNIST_DATA/', train=False, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batchSize, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                          batch_size=batchSize, 
                                          shuffle=False)
# print(net)
loadPath = './Project/param'
appendix = '.t7'
# print(IO.exists(loadPath))

Ensemble_num = 10 # Smaller than batchSize ( number of samples used in each batch )

n = 1   # w index
net = Net()
path = loadPath + str(n) + appendix
model_dict=net.load_state_dict(torch.load(path))
for images, labels in test_loader:
    outputs = net(images)
print(len(FC1)) 

# Dimensionality of FC layers
h_fc1_array = torch.zeros(len(FC1)*Ensemble_num,64)
for i in range(len(FC1)):
    for j in range(Ensemble_num): # channel idx = 1
        h_fc1_array[i*Ensemble_num+j,:] = FC1[i][j].reshape(1,64)
h_fc1_array_T = torch.transpose(h_fc1_array,0,1)
CMatrix = torch.mm(h_fc1_array_T,h_fc1_array)/h_fc1_array.size(0)
h_fc1_array_mean = torch.mean(h_fc1_array,0).view(1,h_fc1_array.size(1))
h_fc1_array_mean_T = torch.transpose(h_fc1_array_mean,0,1)
CMatrix_ = torch.mm(h_fc1_array_mean_T,h_fc1_array_mean)
C = CMatrix - CMatrix_
D = np.square(torch.trace(C).detach().numpy())/torch.trace(torch.mm(C,C)).detach().numpy()
print('Dimensionality of fc layer1 is {0}'.format(D))
