import os
# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import ShapeNet
# import torch_geometric.transforms as T
import numpy as np
import pyvista as pv
from pyvista import examples
import binvox_rw
import matplotlib.pyplot as plt

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

def vox_show(vox, mode='matplot', show_mesh=None):
  # Cut mesh in half
  # for x in range(32):
  #   for y in range(32):
  #     for z in range(32):
  #       if x >= 16:
  #         vox[x][y][z] = 0

  if mode=='pyvista':
    grid = pv.UniformGrid()
    grid.dimensions = np.array(vox.shape) + 1
    grid.origin = (0, 0, 0) 
    grid.spacing = (1, 1, 1)
    grid.cell_data["values"] = vox.flatten(order="F") * 256

    pl = pv.Plotter(lighting='light_kit') # 'three lights'  # 'none'
    pl.add_volume(grid, cmap="viridis", opacity=[(i > 5) * 256 for i in range(256)] )#, opacity="sigmoid"
    # if show_mesh:
    #   pl.add_mesh(show_mesh, color="green", opacity=1.0)
    pl.show()


  if mode=='matplot':
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_axis_off()
    ax.voxels(vox, facecolors='teal') # edgecolor='k', 
    plt.ioff()
    plt.show()

tst_class = '03001627'
size = 64
test_index = 2
vox_root = 'pre_stage_3/'+tst_class
raw_root = 'pre_stage_1/'+tst_class
obj_names = os.listdir(vox_root)
obj_names = sorted(obj_names)
with open(vox_root + '/' + obj_names[test_index] + '/vox' + str(size) + '_depth_fusion_5.binvox', 'rb') as f:
  print(vox_root + '/' + obj_names[test_index])
  raw_vox = binvox_rw.read_as_3d_array(f)
raw_obj = pv.read(raw_root + '/' + obj_names[test_index] + '/model.obj')
print(raw_vox.dims)
vox_data = np.swapaxes(raw_vox.data, 1, 2)

# vox_show(vox_data)
vox_show(vox_data, mode='pyvista', show_mesh=raw_obj)