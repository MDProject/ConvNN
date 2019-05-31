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
        #conv4.append(x) 
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

weightReplica = [] # list of weight tensors
Ensemble_num = 10 # Smaller than batchSize ( number of samples used in each batch )

"""
for n in range(Ensemble_num):
    net = Net()
    path = loadPath + str(n) + appendix
    model_dict=net.load_state_dict(torch.load(path))
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

print("Data loading complete")
"""

# <hihj> matrixxZAc
n = 1   # w index
net = Net()
path = loadPath + str(n) + appendix
model_dict=net.load_state_dict(torch.load(path))
for images, labels in test_loader:
    outputs = net(images)
print(len(conv1))
print(len(conv2)) # 200 
h_act1 = torch.zeros(len(conv1)*Ensemble_num, 24*24)
h_act2 = torch.zeros(len(conv2)*Ensemble_num, 20*20)
h_act3 = torch.zeros(len(conv3)*Ensemble_num, 16*16)
h_act4 = torch.zeros(len(conv4)*Ensemble_num, 12*12)
# h_act5 = torch.zeros(len(conv1)*Ensemble_num, 8*8)

"""
D1 = np.zeros(conv1[0][0].size(0))
for cindex in range(conv1[0][0].size(0)):
    for i in range(len(conv1)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act1[i*Ensemble_num+j,:] = conv1[i][j,cindex].reshape(1,24*24)
    h_act1_T = torch.transpose(h_act1,0,1)
    CMat1 = torch.mm(h_act1_T,h_act1)/h_act1.size()[0]
    h_act1_ = torch.mean(h_act1,0)
    h_act1_ = h_act1_.view(1,h_act1_.size(0))
    CMat1_ = torch.mm(torch.transpose(h_act1_,0,1),h_act1_)
    CovMatrix1 = CMat1 - CMat1_
    D1[cindex] = np.square(torch.trace(CovMatrix1).detach().numpy())/torch.trace(torch.mm(CovMatrix1,CovMatrix1)).detach().numpy()
print(D1)
"""
"""
D2 = np.zeros(conv2[0][0].size(0))
for cindex in range(conv2[0][0].size(0)):
    for i in range(len(conv2)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act2[i*Ensemble_num+j,:] = conv2[i][j,cindex].reshape(1,20*20)
    h_act2_T = torch.transpose(h_act2,0,1)
    CMat2 = torch.mm(h_act2_T,h_act2)/h_act2.size()[0]
    h_act2_ = torch.mean(h_act2,0)
    h_act2_ = h_act2_.view(1,h_act2_.size(0))
    CMat2_ = torch.mm(torch.transpose(h_act2_,0,1),h_act2_)
    CovMatrix2 = CMat2 - CMat2_
    D2[cindex] = np.square(torch.trace(CovMatrix2).detach().numpy())/torch.trace(torch.mm(CovMatrix2,CovMatrix2)).detach().numpy()      
print(D2)
""" 
"""    
D3 = np.zeros(conv3[0][0].size(0))
for cindex in range(conv3[0][0].size(0)):
    for i in range(len(conv3)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act3[i*Ensemble_num+j,:] = conv3[i][j,cindex].reshape(1,16*16)
    h_act3_T = torch.transpose(h_act3,0,1)
    CMat3 = torch.mm(h_act3_T,h_act3)/h_act3.size()[0]
    h_act3_ = torch.mean(h_act3,0)
    h_act3_ = h_act3_.view(1,h_act3_.size(0))
    CMat3_ = torch.mm(torch.transpose(h_act3_,0,1),h_act3_)
    CovMatrix3 = CMat3 - CMat3_
    D3[cindex] = np.square(torch.trace(CovMatrix3).detach().numpy())/torch.trace(torch.mm(CovMatrix3,CovMatrix3)).detach().numpy()       
print(D3)
"""
"""
D4 = np.zeros(conv4[0][0].size(0))
for cindex in range(conv4[0][0].size(0)):
    for i in range(len(conv4)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act4[i*Ensemble_num+j,:] = conv4[i][j,cindex].reshape(1,12*12)
    h_act4_T = torch.transpose(h_act4,0,1)
    CMat4 = torch.mm(h_act4_T,h_act4)/h_act4.size()[0]
    h_act4_ = torch.mean(h_act4,0)
    h_act4_ = h_act4_.view(1,h_act4_.size(0))
    CMat4_ = torch.mm(torch.transpose(h_act4_,0,1),h_act4_)
    CovMatrix4 = CMat4 - CMat4_
    D4[cindex] = np.square(torch.trace(CovMatrix4).detach().numpy())/torch.trace(torch.mm(CovMatrix4,CovMatrix4)).detach().numpy()
print(D4)
"""
print("PAUSE")

n = 1   # w index
net = Net()
print(net)
path = loadPath + str(n) + appendix
model_dict=net.load_state_dict(torch.load(path))
for param_tensor in net.state_dict():
    # print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    if param_tensor == 'conv1.weight':
        we_rep = net.state_dict()[param_tensor]
        weightReplica.append(we_rep)
    if param_tensor == 'conv2.weight':
        we_rep = net.state_dict()[param_tensor]
        weightReplica.append(we_rep)
    if param_tensor == 'conv3.weight':
        we_rep = net.state_dict()[param_tensor]
        weightReplica.append(we_rep)
    if param_tensor == 'conv4.weight':
        we_rep = net.state_dict()[param_tensor]
        weightReplica.append(we_rep)

# plot and save weight distribution
def statisticOfWeight(array_w,min,max,intervals):
    dw = (max - min)/intervals
    freq = np.zeros((intervals))
    segment = np.linspace(min,max,intervals,endpoint=False)
    for w in array_w:
        idx = int(np.floor(((w-min)/dw).item()))
        freq[idx] = freq[idx] + 1
    return freq , segment

Min = -0.4
Max = 0.4
root = './Project/weight/'
"""
# weight distribution of each channel and layer
for l in range(4):
    weight_tensor = weightReplica[l]
    numOfChannel = weight_tensor.size(0)
    weight_tensor = weight_tensor.view(numOfChannel,-1)
    for c in range(numOfChannel):
        fname = root + 'layer' + str(l) + '_' + 'channel' + str(c) + '.jpg' 
        freq,seg = statisticOfWeight(weight_tensor[c,:],Min,Max,40)
        plt.figure()
        plt.plot(seg, freq, '-o')
        plt.savefig(fname)
"""
# weight distribution of each conv*
Wt = []
for l in range(4):
    weight_tensor = weightReplica[l]
    weight_tensor = weight_tensor.view(1,-1)
    Wt.append(weight_tensor[0])
    fname = root + 'Layer' + str(l) + '.jpg' 
    freq,seg = statisticOfWeight(weight_tensor[0],Min,Max,40)
    plt.figure()
    plt.plot(seg, freq, '-o')
    plt.savefig(fname)
Wt = torch.cat((Wt[0],Wt[1],Wt[2],Wt[3]),0)

# weight distribution of all conv
fname = root + 'AllLayer' + '.jpg' 
freq,seg = statisticOfWeight(Wt,Min,Max,40)
plt.figure()
plt.plot(seg, freq, '-o')
plt.savefig(fname)
print('weight distribution saved')
