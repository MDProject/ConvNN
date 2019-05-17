import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os.path as IO
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.conv5 = nn.Conv2d(64, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# print(net)
loadPath = './Project/param'
appendix = '.t7'
# print(IO.exists(loadPath))

weightReplica = [] # list of weight tensors
Ensemble_num = 50
for n in range(Ensemble_num):
    net = Net()
    path = loadPath + str(n) + appendix
    model_dict=net.load_state_dict(torch.load(path))
    for param_tensor in net.state_dict():
        # print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        if param_tensor == 'conv1.weight':
            we_rep = net.state_dict()[param_tensor]
            weightReplica.append(we_rep)

print("Data loading complete")
print("Total {} samples of weight tensor".format(len(weightReplica)))
"""
print(weightReplica[3])
print(weightReplica[2].size())  [6,1,5,5] [outChannel, inChannel, kernal]
"""

# weight distribution of in single channel
size1 = weightReplica[0].size()
Nlayer1 = size1[1]
Nlayer2 = size1[0]
kernel_size = [size1[2],size1[3]]
weight_mu = torch.zeros(kernel_size)
for n in range(Ensemble_num):
    weight_mu = weight_mu + weightReplica[n]
weight_mu = weight_mu/Ensemble_num

# initial standard error to be 0
weight_sigma = torch.zeros(kernel_size)
for n in range(Ensemble_num):
    weight_sigma = weight_sigma + weightReplica[n]*weightReplica[n]
weight_sigma2 = weight_sigma/Ensemble_num
weight_sigma2 = weight_sigma2 - weight_mu*weight_mu
weight_sigma = torch.rsqrt(weight_sigma2)

print("End")
