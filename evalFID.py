'''
evalFID.py from DECOR-GAN's python code (https://github.com/czq142857/DECOR-GAN)
'''

import os
import time
# import math
# import random
import numpy as np
# import h5py
# import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.linalg import sqrtm

import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import normal
# from torch import optim
# from torch.autograd import Variable

import cutils
import binvox_rw_faster as binvox_rw
from torchvision import transforms

from utils import *
import data_loader, model
# import model, data_loader
# import data_loader


max_num_of_styles = 16
max_num_of_contents = 100

random_seed = 10
torch.manual_seed(random_seed)

shapenet_dir = "/local-scratch2/ShapeNetCore.v1.lite/"
#shapenet_dir = "/home/zhiqinc/zhiqinc/ShapeNetCore.v1.lite/"

class FullVoxelShapeNet(Dataset):
  def __init__(self, obj_class_ids, size, fill_type, transform=None):
    self.target_dir = "./pre_stage_3/"
    self.obj_class_ids = obj_class_ids
    self.size = size
    self.fill_type = fill_type
    self.transform = transform
    if not os.path.exists(self.target_dir):
      print("ERROR: this dir does not exist: " + self.target_dir)
      exit()

    self.obj_names = []
    for i in obj_class_ids:
      f = sorted(os.listdir(self.target_dir + i + "/"))
      ids = [i]*len(f)
      self.obj_names += zip(f, ids)
    self.file_name = "/vox"+str(size)+"_"+str(fill_type)+".binvox"

  def __len__(self):
    return len(self.obj_names)

  def __getitem__(self, i):
    if i < len(self.obj_names):
      obj, class_id = self.obj_names[i]
      this_name = self.target_dir + class_id + '/' + obj + self.file_name
      if not os.path.exists(this_name):
        print("Error: File not found:", this_name)
        return

      with open(this_name, 'rb') as voxel_model_file:
        vox_model = binvox_rw.read_as_3d_array(voxel_model_file)
      vox_data = np.swapaxes(vox_model.data, 1, 2)[...,::-1,:].copy()
      if (vox_data.shape[0] == self.size and vox_data.shape[1] == self.size and vox_data.shape[2] == self.size):
        if self.transform:
          vox_data = self.transform(vox_data)
        return vox_data, self.obj_class_ids.index(class_id)
      else:
        print("Error: data in wrong format:", this_name)

# class classifier(nn.Module):
#     def __init__(self, ef_dim, z_dim, class_num, voxel_size):
#         super(classifier, self).__init__()
#         self.ef_dim = ef_dim
#         self.z_dim = z_dim
#         self.class_num = class_num
#         self.voxel_size = voxel_size

#         self.conv_1 = nn.Conv3d(1,             self.ef_dim,   4, stride=2, padding=1, bias=True)
#         self.bn_1 = nn.InstanceNorm3d(self.ef_dim)

#         self.conv_2 = nn.Conv3d(self.ef_dim,   self.ef_dim*2, 4, stride=2, padding=1, bias=True)
#         self.bn_2 = nn.InstanceNorm3d(self.ef_dim*2)

#         self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
#         self.bn_3 = nn.InstanceNorm3d(self.ef_dim*4)

#         self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
#         self.bn_4 = nn.InstanceNorm3d(self.ef_dim*8)

#         self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=2, padding=1, bias=True)

#         if self.voxel_size==256:
#             self.bn_5 = nn.InstanceNorm3d(self.z_dim)
#             self.conv_5_2 = nn.Conv3d(self.z_dim, self.z_dim, 4, stride=2, padding=1, bias=True)

#         self.linear1 = nn.Linear(self.z_dim, self.class_num, bias=True)



#     def forward(self, inputs, is_training=False):
#         out = inputs

#         out = self.bn_1(self.conv_1(out))
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.bn_2(self.conv_2(out))
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.bn_3(self.conv_3(out))
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.bn_4(self.conv_4(out))
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.conv_5(out)

#         if self.voxel_size==256:
#             out = self.bn_5(out)
#             out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
#             out = self.conv_5_2(out)

#         z = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
#         z = z.view(-1,self.z_dim)
#         out = F.leaky_relu(z, negative_slope=0.01, inplace=True)
        
#         out = self.linear1(out)

#         return out, z

class ClassifNetwork(nn.Module):
  def __init__(self, z_dim, class_n):
    super().__init__()
    # hid_1_size = 512
    self.conv_size = 8
    self.z_dim = z_dim
    
    self.conv0 = nn.Conv3d(1, self.conv_size, stride=1, kernel_size=5)
    self.conv1 = nn.Conv3d(self.conv_size, self.conv_size*2, stride=1, kernel_size=5)
    self.conv2 = nn.Conv3d(self.conv_size*2, self.conv_size*4, stride=1, kernel_size=5)
    self.conv3 = nn.Conv3d(self.conv_size*4, self.conv_size*8, stride=1, kernel_size=5)
    self.conv4 = nn.Conv3d(self.conv_size*8, self.conv_size*4, stride=1, kernel_size=5)

    # self.fc1 = nn.Linear(self.conv_size*4 * 4*4*4, latent_n)
    self.fc2 = nn.Linear(z_dim, class_n)
    # self.soft = nn.Softmax(dim=1)

  def forward(self, x):
    x = F.leaky_relu(self.conv0(x))
    x = self.conv1(x)
    x = F.leaky_relu(F.max_pool3d(x, 2))
    x = self.conv2(x)
    # x = F.dropout(x, training=self.training)
    x = F.leaky_relu(F.max_pool3d(x, 2))
    # x = F.leaky_relu(x)
    # x = x.view(-1, self.conv_size*4 * 4*4*4)
    # x = self.fc1(x)
    # x = F.dropout(x, training=self.training)
    # Get the second last perceptual layer
    z = F.adaptive_avg_pool3d(x, output_size=(2,2,2))
    # z = x
    z = z.view(-1, self.z_dim)
    x = F.leaky_relu(z)
    x = self.fc2(x)
    return x, z


def train_classifier(): #config):
  if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
  else:
    device = torch.device('cpu')


  # class_names = os.listdir(shapenet_dir)
  # class_names = [name for name in class_names if name[0]=='0' and len(name)==8]
  # class_num = len(class_names)

  # if config.output_size==128:
  #     voxel_size = 128
  #     batch_size = 32
  # elif config.output_size==256:
  #     voxel_size = 256
  #     batch_size = 4

  z_dim = 256
  num_epochs = 100

  tst_class = '03001627'
  #                 Chair        Plane      Bed         Bench       Bottle       Car        Clock       Cabinet    Headphones  Piano       Rifle        Sofa        Table       Phone     Watercraft   Bathtub
  train_classes = ['03001627', '02691156', '02818832', '02828884', '02876657', '02958343', '03046257', '02933112', '03261776', '03928116', '04090263', '04256520', '04379243', '04401088', '04530566', '02808440']
  voxel_size = 32
  batch_size = 64
  fill_type =  'depth_fusion_5'
  
  #dataloader = torch.utils.data.DataLoader(ShapeNet(shapenet_dir, voxel_size), batch_size=batch_size, shuffle=True, num_workers=16)
  dataloader = torch.utils.data.DataLoader(FullVoxelShapeNet(train_classes, voxel_size, fill_type,
                                                                transform=transforms.Compose([
                                                                          # data_loader.RandomTranslate(2),
                                                                          # data_loader.RandomFlip(0.5)
                                                                          ])),
                                                                batch_size=batch_size, shuffle=True,
                                                                drop_last=True)

  FIDshapenet = ClassifNetwork(z_dim, len(train_classes))
  FIDshapenet.to(device)
  optimizer = torch.optim.Adam(FIDshapenet.parameters(), lr=0.0001, betas=(0.5, 0.999))

  print("start")

  start_time = time.time()
  for epoch in range(num_epochs):

    FIDshapenet.train()

    avg_acc = 0
    avg_count = 0
    for batch in dataloader:
      voxels_, labels_ = batch
      voxels = voxels_[:, None, :, :]
      voxels = voxels.to(device, dtype=torch.float)
      # voxels = voxels_.to(device)
      labels = labels_.to(device).squeeze()

      optimizer.zero_grad()
      pred_labels, _ = FIDshapenet(voxels)
      loss = F.cross_entropy(pred_labels, labels)
      loss.backward()
      optimizer.step()

      acc = torch.mean((torch.max(pred_labels, 1)[1]==labels).float())
      
      avg_acc += acc.item()
      avg_count += 1

    print('Epoch: [%d/%d] time: %.0f, accuracy: %.8f' % (epoch, num_epochs, time.time()-start_time, avg_acc/avg_count))
    torch.save(FIDshapenet.state_dict(), 'evaluation/FIDshapenet_'+str(voxel_size)+'_'+str(fill_type)+'.pth')




# def compute_FID_for_real(config):
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     torch.backends.cudnn.benchmark = True
    # else:
    #     device = torch.device('cpu')

    # # class_num = 24
    # #                 Chair        Plane      Bed         Bench       Bottle       Car        Clock       Cabinet    Headphones  Piano       Rifle        Sofa        Table       Phone     Watercraft   Bathtub
    # train_classes = ['03001627', '02691156', '02818832', '02828884', '02876657', '02958343', '03046257', '02933112', '03261776', '03928116', '04090263', '04256520', '04379243', '04401088', '04530566', '02808440']
    # voxel_size = 32
    # fill_type = 'depth_fusion_5'

    # z_dim = 256

    # # data_dir = config.data_dir

    # #load style shapes
    # fin = open("splits/"+config.data_style+".txt")
    # styleset_names = [name.strip() for name in fin.readlines()]
    # fin.close()
    # styleset_len = len(styleset_names)

    # #load content shapes
    # fin = open("splits/"+config.data_content+".txt")
    # dataset_names = [name.strip() for name in fin.readlines()]
    # fin.close()
    # dataset_len = len(dataset_names)

    # # if config.output_size==128:
    # #     voxel_size = 128
    # # elif config.output_size==256:
    # #     voxel_size = 256

    # FIDshapenet = ClassifNetwork(z_dim, len(train_classes))
    # FIDshapenet.to(device)
    # FIDshapenet.load_state_dict(torch.load('FIDshapenet'+str(voxel_size)+'.pth'))

#     activation_all = np.zeros([dataset_len, z_dim])
#     activation_style = np.zeros([max_num_of_styles, z_dim])

#     target_dir = "./pre_stage_3/"
#     file_name = "/vox"+str(voxel_size)+"_"+str(fill_type)+".binvox"
#     obj_names = []
#     for i in train_classes:
#       f = sorted(os.listdir(target_dir + i + "/"))
#       ids = [i]*len(f)
#       obj_names += zip(f, ids)

#     for content_id in range(dataset_len):
#       print("processing content - "+str(content_id+1)+"/"+str(dataset_len))
#       if content_id < len(obj_names):
#         obj, class_id = obj_names[content_id]
#         this_name = target_dir + obj + file_name
#         if not os.path.exists(this_name):
#           print("Error: File not found:", this_name)
#           continue

#         with open(this_name, 'rb') as voxel_model_file:
#           vox_model = binvox_rw.read_as_3d_array(voxel_model_file)
#         vox_data = np.swapaxes(vox_model.data, 1, 2)[...,::-1,:].copy()
#         if (vox_data.shape[0] == voxel_size and vox_data.shape[1] == voxel_size and vox_data.shape[2] ==voxel_size):
#           tmp_raw = vox_data
#         else:
#           print("Error: data in wrong format:", this_name)
#           continue

#         # if config.output_size==128:
#         #     tmp_raw = get_vox_from_binvox_1over2_return_small(os.path.join(data_dir,dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
#         # elif config.output_size==256:
#         #     tmp_raw = get_vox_from_binvox(os.path.join(data_dir,dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)

#         voxels = torch.from_numpy(tmp_raw).to(device).unsqueeze(0).unsqueeze(0).float()
#         _, z = FIDshapenet(voxels)
#         activation_all[content_id] = z.view(-1).detach().cpu().numpy()


#     for style_id in range(max_num_of_styles):
#       print("processing style - "+str(style_id+1)+"/"+str(styleset_len))

#       if style_id < len(obj_names):
#         obj, class_id = obj_names[content_id]
#         this_name = target_dir + obj + file_name
#         if not os.path.exists(this_name):
#           print("Error: File not found:", this_name)
#           continue

#         with open(this_name, 'rb') as voxel_model_file:
#           vox_model = binvox_rw.read_as_3d_array(voxel_model_file)
#         vox_data = np.swapaxes(vox_model.data, 1, 2)[...,::-1,:].copy()
#         if (vox_data.shape[0] == voxel_size and vox_data.shape[1] == voxel_size and vox_data.shape[2] ==voxel_size):
#           tmp_raw = vox_data
#         else:
#           print("Error: data in wrong format:", this_name)
#           continue

#         # if config.output_size==128:
#         #     tmp_raw = get_vox_from_binvox_1over2_return_small(os.path.join(data_dir,styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
#         # elif config.output_size==256:
#         #     tmp_raw = get_vox_from_binvox(os.path.join(data_dir,styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)

#         voxels = torch.from_numpy(tmp_raw).to(device).unsqueeze(0).unsqueeze(0).float()
#         _, z = FIDshapenet(voxels)
#         activation_style[style_id] = z.view(-1).detach().cpu().numpy()

#     mu_all, sigma_all = np.mean(activation_all, axis=0), np.cov(activation_all, rowvar=False)
#     mu_style, sigma_style = np.mean(activation_style, axis=0), np.cov(activation_style, rowvar=False)

#     hdf5_file = h5py.File("precomputed_real_mu_sigma_"+str(voxel_size)+"_"+config.data_content+"_num_style_"+str(max_num_of_styles)+".hdf5", mode='w')
#     hdf5_file.create_dataset("mu_all", mu_all.shape, np.float32)
#     hdf5_file.create_dataset("sigma_all", sigma_all.shape, np.float32)
#     hdf5_file.create_dataset("mu_style", mu_style.shape, np.float32)
#     hdf5_file.create_dataset("sigma_style", sigma_style.shape, np.float32)

#     hdf5_file["mu_all"][:] = mu_all
#     hdf5_file["sigma_all"][:] = sigma_all
#     hdf5_file["mu_style"][:] = mu_style
#     hdf5_file["sigma_style"][:] = sigma_style
    
#     hdf5_file.close()





# def eval_FID(config):
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         torch.backends.cudnn.benchmark = True
#     else:
#         device = torch.device('cpu')

#     class_num = 24

#     result_dir = "output_for_FID"
#     if not os.path.exists(result_dir):
#         print("ERROR: result_dir does not exist! "+result_dir)
#         exit(-1)

#     output_dir = "eval_output"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     if config.output_size==128:
#         voxel_size = 128
#     elif config.output_size==256:
#         voxel_size = 256

#     z_dim = 512

#     Clsshapenet = classifier(32, z_dim, class_num, voxel_size)
#     Clsshapenet.to(device)
#     Clsshapenet.load_state_dict(torch.load('Clsshapenet_'+str(voxel_size)+'.pth'))

#     num_of_shapes = max_num_of_styles*max_num_of_contents
#     activation_test = np.zeros([num_of_shapes, z_dim])

#     counter = 0
#     for content_id in range(max_num_of_contents):
#         print("processing content - "+str(content_id+1)+"/"+str(max_num_of_contents))
#         for style_id in range(max_num_of_styles):
#             voxel_model_file = open(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox", 'rb')
#             tmp_raw = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
#             if config.output_size==128:
#                 tmp_raw = tmp_raw[64:-64,64:-64,64:-64]

#             voxels = torch.from_numpy(tmp_raw).to(device).unsqueeze(0).unsqueeze(0).float()
#             _, z = Clsshapenet(voxels)
#             activation_test[counter] = z.view(-1).detach().cpu().numpy()
#             counter += 1

#     mu_test, sigma_test = np.mean(activation_test, axis=0), np.cov(activation_test, rowvar=False)

#     hdf5_file = h5py.File("precomputed_real_mu_sigma_"+str(voxel_size)+"_"+config.data_content+"_num_style_"+str(max_num_of_styles)+".hdf5", mode='r')
#     mu_all = hdf5_file["mu_all"][:]
#     sigma_all = hdf5_file["sigma_all"][:]
#     mu_style = hdf5_file["mu_style"][:]
#     sigma_style = hdf5_file["sigma_style"][:]
#     hdf5_file.close()

#     def calculate_fid(mu1, sigma1, mu2, sigma2):
#         ssdiff = np.sum((mu1 - mu2)**2)
#         covmean = sqrtm(sigma1.dot(sigma2))
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real
#         fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#         return fid
    
#     fid_all = calculate_fid(mu_test, sigma_test, mu_all, sigma_all)
#     fid_style = calculate_fid(mu_test, sigma_test, mu_style, sigma_style)

#     #write result_FID_score
#     fout = open(output_dir+"/result_FID.txt", 'w')
#     fout.write("fid_all:\n"+str(fid_all)+"\n")
#     fout.write("fid_style:\n"+str(fid_style)+"\n")
#     fout.close()

# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/#:~:text=Feature%20vectors%20can%20then%20be,*sqrt(C_1*C_2))
# act1/act2 are arrays of z-activations for real and generated data respecitvely
def calc_fid(act1, act2):
  act1 = act1.cpu().detach().numpy()
  act2 = act2.cpu().detach().numpy()
  # calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

def model_FID(model_g_net, tst_class='03001627'):
  # random_seed = 1
  # torch.manual_seed(random_seed)
  # train_classifier()

  if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
  else:
    device = torch.device('cpu')

  # class_num = 24
  #                 Chair        Plane      Bed         Bench       Bottle       Car        Clock       Cabinet    Headphones  Piano       Rifle        Sofa        Table       Phone     Watercraft   Bathtub
  train_classes = ['03001627', '02691156', '02818832', '02828884', '02876657', '02958343', '03046257', '02933112', '03261776', '03928116', '04090263', '04256520', '04379243', '04401088', '04530566', '02808440']
  voxel_size = 32
  fill_type = 'depth_fusion_5'

  z_dim = 256

  FIDshapenet = ClassifNetwork(z_dim, len(train_classes))
  FIDshapenet.to(device)
  FIDshapenet.load_state_dict(torch.load('evaluation/FIDshapenet_'+str(voxel_size)+'_'+str(fill_type)+'.pth'))

  # Load generated data
  n_gen = n_gt = 256
  z = normal.Normal(0.0, 100.0).sample([n_gen, model.latent_n]).to(device)
  g_net = model_g_net.to(device)

  with torch.no_grad():
    gen_vox = g_net(z)

  
  # Load ground truth data
  gt_dataload = torch.utils.data.DataLoader(data_loader.VoxelShapeNet(tst_class, voxel_size, fill_type),
                                                                  batch_size=n_gt, shuffle=True,
                                                                  drop_last=True)
  gt_vox = next(iter(gt_dataload))
  gt_vox = gt_vox[:, None, :, :].to(device, dtype=torch.float)

  _, gen_z = FIDshapenet(gen_vox)
  _, gt_z = FIDshapenet(gt_vox)

  # print("FID score for", tst_class, "is:", calc_fid(gt_z, gen_z))
  return calc_fid(gt_z, gen_z)

def voxel_FID(gen_vox, tst_class='03001627'):
  train_classes = ['03001627', '02691156', '02818832', '02828884', '02876657', '02958343', '03046257', '02933112', '03261776', '03928116', '04090263', '04256520', '04379243', '04401088', '04530566', '02808440']
  voxel_size = 32
  fill_type = 'depth_fusion_5'
  
  if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
  else:
    device = torch.device('cpu')

  
  z_dim = 256
  n_gt = 256

  FIDshapenet = ClassifNetwork(z_dim, len(train_classes))
  FIDshapenet.to(device)
  FIDshapenet.load_state_dict(torch.load('evaluation/FIDshapenet_'+str(voxel_size)+'_'+str(fill_type)+'.pth'))
  
  gt_dataload = torch.utils.data.DataLoader(data_loader.VoxelShapeNet(tst_class, voxel_size, fill_type),
                                                                  batch_size=n_gt, shuffle=True,
                                                                  drop_last=True)
  
  gt_vox = next(iter(gt_dataload))
  gt_vox = gt_vox[:, None, :, :].to(device, dtype=torch.float)
  
  # print(gt_vox.shape)
  # print(torch.from_numpy(np.array([gen_vox])).shape)
  _, gt_z = FIDshapenet(gt_vox)
  gen_vox = torch.from_numpy(np.array([gen_vox])).to(device, dtype=torch.float)
  _, gen_z = FIDshapenet(gen_vox)
  
  return calc_fid(gt_z, gen_z)