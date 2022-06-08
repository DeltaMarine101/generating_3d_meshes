import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import random as r

# Hyperparams
batch_size = 32
epochs = 30
d_lr = 0.0002
g_lr = 0.0002
latent_n = 256

class DiscrimNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    hid_1_size = 128
    self.conv_size = 8
    
    self.conv0 = nn.Conv3d(1, self.conv_size, stride=1, kernel_size=5)
    self.conv1 = nn.Conv3d(self.conv_size, self.conv_size*2, stride=1, kernel_size=5)
    self.conv2 = nn.Conv3d(self.conv_size*2, self.conv_size*4, stride=1, kernel_size=5)
    self.conv3 = nn.Conv3d(self.conv_size*4, self.conv_size*8, stride=1, kernel_size=5)
    self.conv4 = nn.Conv3d(self.conv_size*8, self.conv_size*4, stride=1, kernel_size=5)
    self.conv2_drop = nn.Dropout3d()
    self.fc1 = nn.Linear(self.conv_size*4 * 4*4*4, hid_1_size)
    self.fc2 = nn.Linear(hid_1_size, 1)
    # self.soft = nn.Softmax(dim=1)

  def forward(self, x):
    x = F.relu(self.conv0(x), 2)
    x = self.conv1(x)
    x = F.relu(F.max_pool3d(x, 2))
    x = self.conv2(x)
    x = self.conv2_drop(x)
    x = F.relu(F.max_pool3d(x, 2))
    x = x.view(-1, self.conv_size*4 * 4*4*4)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return torch.sigmoid(x)

class GenNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    hid_1_size = 1024
    self.conv_size = 32
    self.fc2 = nn.Linear(latent_n, hid_1_size)
    self.fc1 = nn.Linear(hid_1_size, self.conv_size*8*8*8)
    # # self.conv2_drop = nn.Dropout3d()
    self.conv0 = nn.Conv3d(self.conv_size,        self.conv_size*2,      stride=1, kernel_size=3, padding=1, dilation=1)
    self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
    self.conv1 = nn.Conv3d(self.conv_size*2,      self.conv_size*2,      stride=1, kernel_size=3, padding=1, dilation=1)
    self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')
    self.conv2 = nn.Conv3d(self.conv_size*2,      self.conv_size,        stride=1, kernel_size=3, padding=1, dilation=1)
    self.conv3 = nn.Conv3d(self.conv_size,        int(self.conv_size/2), stride=1, kernel_size=3, padding=1, dilation=1)
    self.conv4 = nn.Conv3d(int(self.conv_size/2), int(self.conv_size/4), stride=1, kernel_size=3, padding=1, dilation=1)
    self.conv5 = nn.Conv3d(int(self.conv_size/4), 1,                     stride=1, kernel_size=3, padding=1, dilation=1)

  def forward(self, x):
    x = self.fc2(x)
    x = F.relu(self.fc1(x))
    x = x.view(-1, self.conv_size, 8, 8, 8)
    # x = self.up1(x)
    x = F.relu(self.conv0(x))
    x = self.up2(x)
    x = F.relu(self.conv1(x))
    x = self.up3(x)
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.conv5(x)
    return torch.sigmoid(x)

# class d_loss(nn.Module):
#   def __init__(self):
#     super(d_loss, self).__init__()

#   def forward(self, output, target):
#     loss = -1/target.shape[0]*torch.sum((target * torch.log(output) + (1 - target) * torch.log(1 - output)))
#     return loss

# class g_loss(nn.Module):
#   def __init__(self):
#     super(g_loss, self).__init__()

#   def forward(self, output, target):
#     loss = torch.sum((1 - target) * torch.log(1 - output))
#     # print("#####", (1 - target) * torch.log(1 - output))
#     return loss

d_net = DiscrimNetwork()
g_net = GenNetwork() # 10, (28,28))
d_lossFunc = F.binary_cross_entropy # d_loss()
g_lossFunc = F.binary_cross_entropy # g_loss()
d_optimiser = optim.Adam(d_net.parameters(), lr=d_lr)
g_optimiser = optim.Adam(g_net.parameters(), lr=g_lr)