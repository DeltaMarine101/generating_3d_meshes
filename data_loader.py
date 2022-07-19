from torch.utils.data import Dataset
import os, random, time
import numpy as np

import binvox_rw

class VoxelShapeNet(Dataset):
  def __init__(self, obj_class_id, size, type, transform=None):
    self.target_dir = "./pre_stage_3/"+obj_class_id+"/"
    self.size = size
    self.type = type
    self.transform = transform
    if not os.path.exists(self.target_dir):
      print("ERROR: this dir does not exist: " + self.target_dir)
      exit()

    self.obj_names = sorted(os.listdir(self.target_dir))
    self.file_name = "/vox"+str(size)+"_"+str(type)+".binvox"
  
  def __len__(self):
    return len(self.obj_names)

  def __getitem__(self, i):
    if i < len(self.obj_names):
      this_name = self.target_dir + self.obj_names[i] + self.file_name
      if not os.path.exists(this_name):
        print("Error: File not found:", this_name)
        return
      
      with open(this_name, 'rb') as voxel_model_file:
        vox_model = binvox_rw.read_as_3d_array(voxel_model_file)
      vox_data = np.swapaxes(vox_model.data, 1, 2)[...,::-1,:].copy()
      if (vox_data.shape[0] == self.size and vox_data.shape[1] == self.size and vox_data.shape[2] == self.size):
        if self.transform:
          vox_data = self.transform(vox_data)
        return vox_data
      else:
        print("Error: data in wrong format:", this_name)


class RandomTranslate(object):
  def __init__(self, translate_size):
    self.translate_size = translate_size

  def __call__(self, vox):
    sz = vox.shape[0]
    dx = np.random.randint(-self.translate_size, self. translate_size)
    dx_n = abs(dx) if dx < 0 else 0
    dx_p = dx if dx > 0 else 0
    dy = np.random.randint(-self.translate_size, self.translate_size)
    dy_n = abs(dy) if dy < 0 else 0
    dy_p = dy if dy > 0 else 0
    dz = np.random.randint(-self.translate_size, self.translate_size)
    dz_n = abs(dz) if dz < 0 else 0
    dz_p = dz if dz > 0 else 0
    vox = np.pad(vox, ((dx_p, dx_n),(dy_p, dy_n),(dz_p, dz_n)), mode='constant')[dx_n:(-dx_p,sz+dx_n)[dx_p==0],dy_n:(-dy_p,sz+dy_n)[dy_p==0],dz_n:(-dz_p,sz+dz_n)[dz_p==0]]
  
    return vox

class RandomFlip(object):
  def __init__(self, chance):
    self.chance = chance

  def __call__(self, vox):
    if random.random() <= self.chance:
      vox = np.flip(vox, axis=0).copy()

    return vox

class SymmetricAxes(object):
  def __init__(self, axis):
    self.axis = axis

  def __call__(self, vox):
    size = vox.shape[self.axis]
    if self.axis == 0:
      vox[size // 2:,:,:] = 0.0
    if self.axis == 1:
      vox[:,size // 2:,:] = 0.0
    if self.axis == 2:
      vox[:,:,size // 2:] = 0.0
    return vox