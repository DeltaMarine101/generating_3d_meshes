import torch, time, shutil, os, psutil
from torch.distributions import normal
from torchvision import transforms
import pylab as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from scipy import ndimage

import model, data_loader, evalFID
from util_functions import print_progress_bar, show_voxels
import binvox_rw_faster

device = torch.device('cpu')

random_seed = 11
torch.manual_seed(random_seed)

g_net = model.g_net.to(device)
g_net.load_state_dict(torch.load('results/img_res_27/saved_G_model.pth'))

data_res = 32
gen_file = "./generated_results/"
n_generated = 2000
batch_sz = 50
oversample_ratio = 2.0
### Generate 
def main():
  n = 0
  z_vals = np.array([])
  for b in range(n_generated // 50):
    z = normal.Normal(0.0, 10.0).sample([int(batch_sz), model.latent_n]) # .to(device)
    
    z_vals = np.append(z_vals, np.array(z.cpu().detach().numpy()))
    
    for param in g_net.parameters():      # Freeze generator params
        param.requires_grad = False

    gen_vox = g_net(z)
    
    for param in g_net.parameters():      # Unfreeze generator params
      param.requires_grad = True
    
    # FID based sorting dosen't work
    # print("Sorting...")
    # gen_vox = sorted(gen_vox, key=n_FID_avg)

    # for i in gen_vox:
    #   print(n_FID_avg(i))

    for i in tqdm(range(len(gen_vox)), desc=str(b) + '/' + str(n_generated // 50)):
      # print_progress_bar(n, n_generated)
      gen_vox_i = gen_vox[i].reshape((data_res, data_res, data_res))
      gen_vox_i = symmetry_unwrap(gen_vox_i, 0)
      gen_vox_i = np.swapaxes(clean_vox(gen_vox_i), 0, 1)

      out_size = 512
      gen_vox4 = np.zeros((out_size, out_size, out_size))
      # repeat_arr = np.ones(data_res) * int(256 / data_res)
      # gen_vox4 = np.repeat(gen_vox4, repeat_arr.astype('uint8'), axis=1).reshape((256, 256, data_res))
      stride = out_size // data_res
      for x in range(gen_vox_i.shape[0]):
        for y in range(gen_vox_i.shape[1]):
          for z in range(gen_vox_i.shape[2]):
            fill = 0.0
            if gen_vox_i[x, y, z]:
              fill = 1.0
            gen_vox4[x*stride:x*stride+stride, y*stride:y*stride+stride, z*stride:z*stride+stride] = fill
              
      # gen_vox4 = ndimage.zoom(gen_vox_i, (4., 4., 4.)) #, mode='nearest', grid_mode=True)
      # gen_vox4 = clean_vox(gen_vox4)
      
      # print(evalFID.voxel_FID(gen_vox[i]))

      if not os.path.exists(gen_file + 'res_' + str(i + b*batch_sz)):
        os.mkdir(gen_file + 'res_' + str(i + b*batch_sz))
      if i + b == 0: f = open("gen_content.txt", 'w')
      else: f = open("gen_content.txt", 'a')
      f.write('res_' + str(i + b*batch_sz) + '\n') # ../../../generating_3d_meshes/generated_results/
      f.close()
      binvox_rw_faster.write_voxel(gen_vox4, gen_file + 'res_' + str(i + b*batch_sz) + '/model_depth_fusion.binvox', 0)
      g_vox = torch.from_numpy(np.array([[gen_vox_i]])).to(device, dtype=torch.float)
      gen_vox4 = torch.from_numpy(np.array([[gen_vox4]])).to(device, dtype=torch.float)
      show_voxels(g_vox, shape=(1, 1), save=True, out_loc=gen_file + 'res_' + str(i + b*batch_sz) + '/', name='render', size=128)
      # show_voxels(gen_vox4, shape=(1, 1), save=True, out_loc=gen_file + 'res_high_' + str(i + b*batch_sz) + '/', name='render_new', size=512)
      # show_voxels(g_vox, shape=(1, 1), save=True, out_loc=gen_file + 'res_high_' + str(i + b*batch_sz) + '/', name='render_low', size=512)
      # fp.close()
      n += 1
      
  # Save latent vecs
  np.savez("latent_vecs.npz", z_vals)
  
def n_FID_avg(val):
  n = 8
  total = 0
  for i in range(n):
    total += evalFID.voxel_FID(val)
  return total / n

def symmetry_unwrap(vox, axis):
  vox = np.array(vox)
  flipped = np.flip(vox, axis=axis).copy()
  size = vox.shape[axis]
  if axis == 0:
    # print(vox[size//2:,:,:].shape, flipped[:size//2,:,:].shape)
    vox[size//2:,:,:] = flipped[size//2:,:,:]
  if axis == 1:
    vox[:,size//2:,:] = flipped[:,size//2:,:]
  if axis == 2:
    vox[:,:,size//2:] = flipped[:,:,size//2:]
  return vox

def clean_vox(vox):

  vox = np.round(vox)
  # vox = ndimage.binary_closing(vox)
  # vox = ndimage.binary_opening(vox)
   
  vox = ndimage.binary_dilation(vox)
  vox = ndimage.binary_erosion(vox)
  # vox = ndimage.binary_dilation(vox)
  
  # mask = 1.0 - ndimage.white_tophat(input=vox, size=1)
  # vox = mask * vox

  return vox

if __name__ == '__main__':
    main()