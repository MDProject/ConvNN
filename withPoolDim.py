import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os.path as IO
import numpy as np

conv1 = []
conv2 = []
conv3 = []
FC = []
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 5, 5) # 28*28 -- 24*24 + pooling 24*24 -- 12*12
        self.conv2 = nn.Conv2d(5, 10, 5) # 12*12 -- 8*8
        self.conv3 = nn.Conv2d(10, 20, 5) # 8*8 -- 4*4
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        tmp = F.relu(self.conv1(x))
        x = F.avg_pool2d(tmp,(2,2))
        conv1.append(tmp)
        x = F.relu(self.conv2(x))
        conv2.append(x)
        x = F.relu(self.conv3(x))
        conv3.append(x)
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

net = Net()
for layer in list(net._modules.items()):
    print(layer[1])

"""
def for_hook(module, input, output):
    print(module)
    for val in input:
        print("input val:",val.size())
    for out_val in output:
        print("output val:", out_val.size())
"""
# net.conv2.register_forward_hook(for_hook)
# net.conv3.register_forward_hook(for_hook)
batchSize = 50
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# print(IO.exists('../MNIST_DATA/train-labels-idx1-ubyte'))
test_set = dset.MNIST('../MNIST_DATA/', train=False, transform=trans, download=True)
# img, label = train_set[0]

test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                          batch_size=batchSize, 
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=1e-3)
loadPath = './Project/Pool/param'
appendix = '.t7'
#   print(IO.exists(savePath))
loss_bound = 0.01
Ensemble_num = 10
n = 0   # w index
net = Net()
path = loadPath + str(n) + appendix
model_dict=net.load_state_dict(torch.load(path))
for images, labels in test_loader:
    outputs = net(images)
print(len(conv1))
print(len(conv2)) # 200 
h_act1 = torch.zeros(len(conv1)*Ensemble_num, 24*24)
h_act2 = torch.zeros(len(conv2)*Ensemble_num, 8*8)
h_act3 = torch.zeros(len(conv3)*Ensemble_num, 4*4)

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
            h_act2[i*Ensemble_num+j,:] = conv2[i][j,cindex].reshape(1,8*8)
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
            h_act3[i*Ensemble_num+j,:] = conv3[i][j,cindex].reshape(1,4*4)
    h_act3_T = torch.transpose(h_act3,0,1)
    CMat3 = torch.mm(h_act3_T,h_act3)/h_act3.size()[0]
    h_act3_ = torch.mean(h_act3,0)
    h_act3_ = h_act3_.view(1,h_act3_.size(0))
    CMat3_ = torch.mm(torch.transpose(h_act3_,0,1),h_act3_)
    CovMatrix3 = CMat3 - CMat3_
    D3[cindex] = np.square(torch.trace(CovMatrix3).detach().numpy())/torch.trace(torch.mm(CovMatrix3,CovMatrix3)).detach().numpy()       
print(D3)