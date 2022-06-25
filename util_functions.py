import pyvista as pv
import math, os
import numpy as np

# Print iterations progress
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if iteration <= total:
      print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Show multiple voxel results together
def show_voxels(voxs, shape=None, save=False, threshhold=1):
  if shape == None:
    ax_l = math.ceil(math.sqrt(len(voxs)))
    shape = (ax_l, ax_l)

  grids = []
  for i in range(len(voxs)):
    grid = pv.UniformGrid()
    grid.dimensions = np.array(voxs[i].shape[1:]) + 1
    grid.origin = (0, 0, 0) 
    grid.spacing = (1, 1, 1)
    np_vox = voxs[i].cpu().detach().numpy().flatten(order="F") * 256
    grid.cell_data["values"] = np_vox.astype(int)
    grids += [grid]

  total = 0
  pl = pv.Plotter(shape=shape, lighting='light_kit', off_screen=save) # 'three lights'  # 'none'
  pl.set_background('#DDDDDD')
  for x in range(shape[0]):
    for y in range(shape[1]):
      pl.subplot(x, y)
      pl.add_volume(grids[total], cmap="viridis", opacity=[(i > threshhold) * 256 for i in range(256)] )#, opacity="sigmoid"
      total += 1
      if total >= len(grids):
        break
    if total >= len(grids):
      break
  # if show_mesh:
  #   pl.add_mesh(show_mesh, color="green", opacity=1.0)
  if save:
    if not os.path.exists('results/img_res/'):
      os.makedirs('results/img_res/')
    pl.show(screenshot='results/img_res/'+str(save)+'_saved_img.png')
  else:
    pl.show()

