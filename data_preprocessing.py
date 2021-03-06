import os, shutil, binvox_rw_customized, cutils, psutil
import numpy as np
import pyvista as pv
from multiprocessing import Process

# Load and simplify shapenet v2 objects from a class
def load_shapenet(obj_class_id, force_redo=False):
  source_root = "./ShapeNetCore.v2/"+obj_class_id+"/"
  if not os.path.exists(source_root):
    print("ERROR: this dir does not exist: "+source_root)
    exit()

  target_dir = "./pre_stage_1/"+obj_class_id+"/"

  obj_names = os.listdir(source_root)
  obj_names = sorted(obj_names)

  def load_obj(dire):
    # print(dire)
    fin = open(dire,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = []
    
    for i in range(len(lines)):
      line = lines[i].split()
      if len(line)==0:
        continue
      if line[0] == 'v':
        x = float(line[1])
        y = float(line[2])
        z = float(line[3])
        vertices.append([x,y,z])
      if line[0] == 'f':
        x = int(line[1].split("/")[0])
        y = int(line[2].split("/")[0])
        z = int(line[3].split("/")[0])
        triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)
    
    #remove isolated points
    vertices_mapping = np.full([len(vertices)], -1, np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vertices_mapping[triangles[i,j]] = 1
    counter = 0
    for i in range(len(vertices)):
        if vertices_mapping[i]>0:
            vertices_mapping[i] = counter
            counter += 1
    vertices = vertices[vertices_mapping>=0]
    triangles = vertices_mapping[triangles]
    
    
    #normalize diagonal=1
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    
    return vertices, triangles

  def write_obj(dire, vertices, triangles):
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
      fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
      fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()

  err_n = 0
  for i in range(len(obj_names)):
    this_name = source_root + obj_names[i] + "/models/model_normalized.obj"
    vox128_name = source_root + obj_names[i] + "/models/model_normalized.solid.binvox"
   
    print('Stage1', str(i)+'/'+str(len(obj_names)), this_name)
    out_dir = target_dir + obj_names[i]
    out_name = out_dir + "/model.obj"
    # Grab the 128^3 binvox file already included in shapenet
    vox_out_dir = "./pre_stage_2/" + obj_class_id + "/" + obj_names[i]
    vox_out_name = vox_out_dir + "/vox128.binvox"

    if os.path.exists(out_name) and os.path.exists(vox_out_name):
      continue

    exists = True
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      os.makedirs(vox_out_dir)
      exists = False
    if (force_redo and exists) or not exists:
      try:
        v,t = load_obj(this_name)
        write_obj(out_name, v,t)

        shutil.copyfile(vox128_name, vox_out_name)
      except UnicodeDecodeError:
        print("Unicode error in file", this_name)
        err_n += 1
      except FileNotFoundError:
        print("File not found error", this_name)
        err_n += 1

  print(len(obj_names) - err_n, "files processed successfully.  ", err_n, "errors occured.")



# Function to voxelise simplified data
def voxelise_data(obj_class_id, size=512):
  target_dir = "./pre_stage_1/"+obj_class_id+"/"
  vox_out_dir = "./pre_stage_2/" + obj_class_id + "/"
  if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

  obj_names = os.listdir(target_dir)
  obj_names = sorted(obj_names)


  for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model.obj"
    vox_gen_loc = target_dir + obj_names[i]
    vox_out_loc = vox_out_dir + obj_names[i]
    vox_out_name = vox_out_loc + "/vox"+str(size)+".binvox"
    print('Stage2', str(i)+'/'+str(len(obj_names)), this_name)
    if os.path.exists(vox_out_name):
      continue

    maxx = 0.5
    maxy = 0.5
    maxz = 0.5
    minx = -0.5
    miny = -0.5
    minz = -0.5

    command = "binvox.exe -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "+" -d "+str(size)+" -e "+this_name+" >> ./null_log.txt"
    
    # if(os.path.isfile(vox_gen_loc + "/model.binvox")):
    #   os.remove(vox_gen_loc + "/model.binvox")

    os.system(command)
    if (os.path.isfile(vox_gen_loc + "/model.binvox")):
      shutil.copyfile(target_dir + obj_names[i] + "/model.binvox", vox_out_name)
      os.remove(target_dir + obj_names[i] + "/model.binvox")


# Function to fill the hollow centers of the voxelised meshes generated by binvox
def fill_voxels(obj_class_id, size=512, type='depth_fusion_5'):
  target_dir = "./pre_stage_2/"+obj_class_id+"/"
  vox_out_dir = "./pre_stage_3/" + obj_class_id + "/"
  if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

  obj_names = os.listdir(target_dir)
  obj_names = sorted(obj_names)

  

  if type == 'depth_fusion_5':
    rendering = np.zeros([2560,2560,5], np.int32)
    state_ctr = np.zeros([512*512*64,2], np.int32)

    for i in range(len(obj_names)):
      this_name = target_dir + obj_names[i] + "/vox"+str(size)+".binvox"
      out_name = vox_out_dir + obj_names[i] + "/vox"+str(size)+"_depth_fusion_5.binvox"
      print('Stage3',str(i)+'/'+str(len(obj_names)), type, this_name)
      if os.path.exists(out_name):
        continue

      try:
        if not os.path.exists(this_name):
          print("File not found:", this_name)
          continue
        voxel_model_file = open(this_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)
        rendering[:] = 2**16
        cutils.depth_fusion_XZY_5views(batch_voxels,rendering,state_ctr)

        if not os.path.exists(vox_out_dir + obj_names[i]):
          os.makedirs(vox_out_dir + obj_names[i])
        with open(out_name, 'wb+') as fout:
          binvox_rw_customized.write(vox_model, fout, state_ctr)
      except:
        print("ERROR")

  elif type == 'depth_fusion_17':
    rendering = np.zeros([2560,2560,17], np.int32)
    state_ctr = np.zeros([512*512*64,2], np.int32)

    for i in range(len(obj_names)):
      this_name = target_dir + obj_names[i] + "/vox"+str(size)+".binvox"
      out_name = vox_out_dir + obj_names[i] + "/vox"+str(size)+"_depth_fusion_17.binvox"
      print('Stage3',str(i)+'/'+str(len(obj_names)), type, this_name)
      if os.path.exists(out_name):
        continue

      try:
        voxel_model_file = open(this_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)
        rendering[:] = 2**16
        cutils.depth_fusion_XZY(batch_voxels,rendering,state_ctr)

        if not os.path.exists(vox_out_dir + obj_names[i]):
            os.makedirs(vox_out_dir + obj_names[i])
        with open(out_name, 'wb+') as fout:
          binvox_rw_customized.write(vox_model, fout, state_ctr)
      
      except:
          print("ERROR")

  elif type == 'floodfill':
    queue = np.zeros([512*512*64,3], np.int32)
    state_ctr = np.zeros([512*512*64,2], np.int32)

    for i in range(len(obj_names)):
      this_name = target_dir + obj_names[i] + "/vox"+str(size)+".binvox"
      out_name = vox_out_dir + obj_names[i] + "/vox"+str(size)+"_filled.binvox"
      print('Stage3',str(i)+'/'+str(len(obj_names)), type, this_name)
      if os.path.exists(out_name):
        continue

      try:
        voxel_model_file = open(this_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)+1
        cutils.floodfill(batch_voxels,queue,state_ctr)

        if not os.path.exists(vox_out_dir + obj_names[i]):
          os.makedirs(vox_out_dir + obj_names[i])
        with open(out_name, 'wb+') as fout:
          binvox_rw_customized.write(vox_model, fout, state_ctr)
      
      except:
          print("ERROR")

  else:
    print("Wrong type entered!")

def process_class_stg1(tst_class):
  p = psutil.Process(os.getpid())
  p.nice(psutil.HIGH_PRIORITY_CLASS)

  load_shapenet(tst_class)

def process_class_stg2(tst_class):
  p = psutil.Process(os.getpid())
  p.nice(psutil.HIGH_PRIORITY_CLASS)

  voxelise_data(tst_class, 16)
  # voxelise_data(tst_class, 32)
  # voxelise_data(tst_class, 64)
  # # voxelise_data(tst_class, 128) # included in shapenet.v2
  # voxelise_data(tst_class, 256)
  # voxelise_data(tst_class, 512)

def process_class_stg3(tst_class):
  # p = psutil.Process(os.getpid())
  # p.nice(psutil.HIGH_PRIORITY_CLASS)

  for type in ['depth_fusion_5', 'depth_fusion_17', 'floodfill']:
    for size in [16]:#[32, 64, 128, 256]:#, 512]:
      print(type, size, "###################################################################")
      fill_voxels(tst_class, type=type, size=size)

def main():
  #                 Chair        Plane      Bed         Bench       Bottle       Car        Clock       Cabinet    Headphones  Piano       Rifle        Sofa        Table       Phone     Watercraft   Bathtub
  train_classes = ['03001627', '02691156', '02818832', '02828884', '02876657', '02958343', '03046257', '02933112', '03261776', '03928116', '04090263', '04256520', '04379243', '04401088', '04530566', '02808440']

  # multiprocessing multiple classes
  # train_classes = train_classes[1:] # these are already processed
  # pl = []
  # for i in train_classes:
  #   p = Process(target=process_class_stg1, args=(i, ))
  #   p.start()
  #   pl += [p]
  # for i in pl:
  #   i.join()

  # pl = []
  # for i in train_classes:
  #   p = Process(target=process_class_stg2, args=(i, ))
  #   p.start()
  #   pl += [p]
  # for i in pl:
  #   i.join()


  # # Load and simplify shapenet (based on DECOR-GAN code) and load the included binvox files
  # p = psutil.Process(os.getpid())
  # p.nice(psutil.HIGH_PRIORITY_CLASS)
  # for i in train_classes:
  #   tst_class = i
  tst_class = train_classes[0]
  #   load_shapenet(tst_class)
  voxelise_data(tst_class, 16)
  #   voxelise_data(tst_class, 32)
  #   voxelise_data(tst_class, 64)
  #   # voxelise_data(tst_class, 128) # included in shapenet.v2
  #   voxelise_data(tst_class, 256)
  #   voxelise_data(tst_class, 512)
  for type in ['depth_fusion_5', 'depth_fusion_17', 'floodfill']:
    for size in [16]: # [32, 64, 128, 256, 512]:
      print(type, size, "###################################################################")
      fill_voxels(tst_class, type=type, size=size)

  # pl = []
  # for i in train_classes:
  #   p = Process(target=process_class_stg3, args=(i, ))
  #   p.start()
  #   pl += [p]
  # for i in pl:
  #   i.join()
  # tst_class =  i

if __name__ == '__main__':
  main()