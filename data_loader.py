from torch.utils.data import Dataset
import os
import numpy as np

import binvox_rw

class VoxelShapeNet(Dataset):
  def __init__(self, obj_class_id, size, type, transform=None):
    self.target_dir = "./pre_stage_3/"+obj_class_id+"/"
    self.size = size
    self.type = type
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
        return vox_data
      else:
        print("Error: data in wrong format:", this_name)
