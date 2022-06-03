# >> source venv/scripts/activate
# >> /usr/bin/env "c:\\Users\\Delta\\GoogleDrive\\University_Work\\Research\\Honours thesis\\Thesis code\\generating_3d_meshes\\venv\\Scripts\\python.exe"

# from tkinter import W
import torch, os
from torch import Tensor
# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import ShapeNet
# import torch_geometric.transforms as T
import numpy as np
import pyvista as pv
from pyvista import examples
import binvox_rw
import matplotlib.pyplot as plt

# SHAPENET_PATH = "./ShapeNetCore.v2"
# shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2, load_textures=False)
# print(len(shapenet_dataset), shapenet_dataset[0])

# def to_pyvista(obj):
#   verts = np.array(obj['verts'])
#   faces = np.array(obj['faces'])
#   h, w = faces.shape
#   b = np.full((h,w+1), 3)
#   b[:,:-1] = faces
#   return pv.PolyData(verts, np.hstack(b))

# surf = to_pyvista(shapenet_dataset[0])
# print(surf)

bunny = examples.download_bunny_coarse()
# print(bunny)

# dataset = ShapeNet(root='.', categories=["Airplane", "Car", "Mug"], include_normals=False)

cpos = [
  (0.05968545747614732, 0.21499133662987105, 0.844081123601417),
  (0.004019019626409824, -0.005962922065511054, -0.08714371802268502),
  (-0.022236440006623177, 0.9730434313769921, -0.22954742732150243)
]

# bunny.plot(cpos=cpos, opacity=0.75)

pv.set_plot_theme("document")

      # Custom voxelise algorithm probably needed
      # Use 'alg = _vtk.vtkSelectEnclosedPoints()'

def vox_conv_show(model):
  model.plot()
  vox_size = 32
  voxels = pv.voxelize(model, density=model.length / vox_size, check_surface=False)
  print("TEST")
  # voxels.compute_implicit_distance(model, inplace=True) # Distance from suface
  print(voxels)
  pl = pv.Plotter(lighting='light_kit') # 'three lights'  # 'none'
  pl.add_mesh(voxels, color='lightblue', show_edges=True, edge_color='black', line_width=2.0, opacity=.90) # , scalars="implicit_distance"
  # pl.add_mesh(bunny, color="red", opacity=0.5)
  pl.show(cpos=cpos)

# vox_conv_show(bunny)

def vox_show(vox):
  print(vox)
  voxels = pv.UniformGrid()
  voxels.dimensions = np.array(vox.shape) + 1
  voxels.origin = (0, 0, 0)
  voxels.spacing = (1, 1, 1)
  voxels.cell_data["values"] = vox.flatten(order="F")

  pl = pv.Plotter(lighting='light_kit') # 'three lights'  # 'none'
  pl.add_mesh(voxels, color='lightblue', show_edges=True, edge_color='black', line_width=2.0, opacity=.90) # , scalars="implicit_distance"
  # pl.add_mesh(bunny, color="red", opacity=0.5)
  pl.show(cpos=cpos)

tst_class = '03001627'
size = 32
test_index = 2
src_root = 'pre_stage_3/'+tst_class
obj_names = os.listdir(src_root)
obj_names = sorted(obj_names)
with open(src_root + '/' + obj_names[test_index] + '/vox' + str(size) + '_depth_fusion_5.binvox', 'rb') as f:
  print(src_root + '/' + obj_names[test_index])
  m1 = binvox_rw.read_as_3d_array(f)
print(m1.dims)

ax = plt.figure().add_subplot(projection='3d')
ax.set_axis_off()
vox_data = np.swapaxes(m1.data, 1, 2)

# Check inside of mesh
for x in range(32):
  for y in range(32):
    for z in range(32):
      if x >= 16:
        vox_data[x][y][z] = 0

ax.voxels(vox_data, facecolors='teal') # edgecolor='k', 
plt.ioff()
plt.show()
# vox_show(m1.data)

# pv.save_meshio('bunny.obj', bunny) 

# >> ./binvox.exe -d 64 -t vtk bunny.obj
# binvox = pv.read('bunny.vtk')
# print(binvox)
# pl2 = pv.Plotter(lighting='light_kit') # 'three lights'  # 'none'
# pl2.add_volume(binvox, cmap="magma")#, opacity="sigmoid"
# pl2.show()
# f = open("tmp.txt", "a")
# f.write(str(binvox['voxel_data'].tostring()))
# f.close()