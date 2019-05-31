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

conv1 = []
conv2 = []
conv3 = []
conv4 = []
conv5 = []
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
        conv4.append(x) 
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

# <hihj> of different channels
n = 1   # w index
net = Net()
path = loadPath + str(n) + appendix
model_dict=net.load_state_dict(torch.load(path))
for images, labels in test_loader:
    outputs = net(images)
print(len(conv4))

h_act4 = []
for cindex in range(conv4[0][0].size(0)):
    h_act4_array = torch.zeros(len(conv4)*Ensemble_num,12*12)
    for i in range(len(conv4)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act4_array[i*Ensemble_num+j,:] = conv4[i][j,cindex].reshape(1,12*12)
    h_act4.append(h_act4_array)
idx_i = [0,1,2,3,4,5,6,7,8,9,1,1,1,1,1,2,2,2,5,5,5,8,8,8]
idx_j = [0,1,2,3,4,5,6,7,8,9,2,3,5,7,9,4,6,8,1,3,9,1,5,7]
root = './Project/OffDiag/'
for i in range(len(idx_i)):
    h_act_i = h_act4[idx_i[i]]
    h_act_j = h_act4[idx_j[i]]
    h_act_i_T = torch.transpose(h_act_i,0,1)
    CMatrix = torch.mm(h_act_i_T,h_act_j)/h_act_i.size(0)
    h_act_i_mean = torch.mean(h_act_i,0)
    h_act_j_mean = torch.mean(h_act_j,0)
    h_act_i_mean = h_act_i_mean.view(1,h_act_i_mean.size(0))
    h_act_j_mean = h_act_j_mean.view(1,h_act_j_mean.size(0))
    CMatrix_ = torch.mm(torch.transpose(h_act_i_mean,0,1),h_act_j_mean)
    C = (CMatrix - CMatrix_).detach().numpy()
    fname = root + 'layer4_' + str(idx_i[i]) + '_' + str(idx_j[i]) + '.jpg'
    plt.figure()
    plt.imshow(C)  
    plt.colorbar()
    plt.savefig(fname) 




