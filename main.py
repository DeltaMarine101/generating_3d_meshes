# >> source venv/scripts/activate
# >> /usr/bin/env "c:\\Users\\Delta\\GoogleDrive\\University_Work\\Research\\Honours thesis\\Thesis code\\generating_3d_meshes\\venv\\Scripts\\python.exe"

import torch, time, shutil, os, psutil
from torch.distributions import normal
from torchvision import transforms
import pylab as plt
import numpy as np
import matplotlib
from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk

import model, data_loader, evalFID
from util_functions import print_progress_bar, show_voxels

# High priority for background performance
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

tst_class = '03001627'
data_res = 32
random_seed = 1
d_ratio = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(random_seed)

trainset = torch.utils.data.DataLoader(data_loader.VoxelShapeNet(tst_class, data_res, 'depth_fusion_5',
                                                                  transform=transforms.Compose([
                                                                            #data_loader.RandomTranslate(2),
                                                                            data_loader.RandomFlip(0.5)
                                                                            ])),
                                                                  batch_size=model.batch_size, shuffle=True,
                                                                  drop_last=True)

# Main
def main():
  if torch.cuda.is_available() and device !=  torch.device('cpu'):
    print("Using device: {}".format(str(torch.cuda.get_device_name(device))))
    print("Cuda version:", torch.version.cuda)
  else:
     print("Using device: {}".format(str(device)))
  
  if not os.path.exists('results/img_res/'):
    os.makedirs('results/img_res/')
  shutil.copy('model.py', 'results/img_res/model_cpy.py')
  shutil.copy('main.py', 'results/img_res/main_cpy.py')

  d_net = model.d_net.to(device)
  d_criterion = model.d_lossFunc
  d_optimiser = model.d_optimiser

  g_net = model.g_net.to(device)
  g_criterion = model.g_lossFunc
  g_optimiser = model.g_optimiser

  plt.ion()
  plt.rcParams['toolbar'] = 'None'
  X = np.linspace(0, model.epochs-1, model.epochs)
  Y = [0]*model.epochs
  # mask = np.ma.masked_where(Y > 0.7, Y)
  graph = plt.plot(Y)[0]
  plt.ylim(0, 10000)
  fig = plt.gcf()

  times = []
  fid = np.array([])
  for epoch in range(1, model.epochs+1):
    total_g_loss = 0
    total_d_loss = 0
    total_vox = 0
    # total_correct = 0

    t0 = time.time()
    batch_n = 0
    print()
    print_progress_bar(0, len(trainset)-1, prefix = 'Epoch '+ str(epoch) + '/' + str(model.epochs) +' Progress:', suffix = 'Complete', length = 50)
    for batch in trainset:                # Load batch
      batch_n += 1
      z = normal.Normal(0.0, 100.0).sample([model.batch_size, model.latent_n]).to(device)

      gen_vox = g_net(z)

      vox = batch[:, None, :, :]
      vox = vox.to(device, dtype=torch.float)

      # Show training data (slow)
      # show_voxels(vox, save='gt')

      preds_real = d_net(vox)            # Process discriminator batch
      preds_fake = d_net(gen_vox)
      
      # Train discriminator
      d_loss = 0
      if batch_n % d_ratio == 0:
        l1 = d_criterion(preds_real,  torch.ones(model.batch_size, 1).to(device) - 0.01)
        l2 = d_criterion(preds_fake,  torch.zeros(model.batch_size, 1).to(device) + 0.01)
        d_loss = (l1 + l2) / 2   # Calculate loss

        for param in g_net.parameters():      # Freeze generator params
          param.requires_grad = False

        d_optimiser.zero_grad()
        d_loss.backward(retain_graph=True)    # Calculate gradients

        for param in g_net.parameters():      # Unfreeze generator params
          param.requires_grad = True

      # Train generator
      g_loss = g_criterion(preds_fake, torch.ones(model.batch_size, 1).to(device))   # Calculate loss
      g_optimiser.zero_grad()
      g_loss.backward()                     # Calculate gradients
      
      d_optimiser.step()                    # Update weights
      g_optimiser.step()

      total_vox += vox.size(0)
      total_g_loss += g_loss
      total_d_loss += d_loss

      fig.canvas.draw()
      plt.pause(0.001)

      print_progress_bar(batch_n, len(trainset)-1, prefix = 'Epoch '+ str(epoch) + '/' + str(model.epochs) +' Progress:', suffix = 'Complete', length = 50)

    # model_accuracy = total_correct / total_images * 100
    print('epoch {0} | G loss: {2:.4f} | D loss: {2:.4f}'.format(epoch, total_g_loss / total_vox, total_d_loss / total_vox))

    # show and save FID score
    fid = np.append(fid, [evalFID.model_FID(g_net)])
    print('FID score:', round(fid[-1], 2), '(min:', str(round(np.min(fid))) + ')')
    f = open('results/img_res/fid_score.txt', ('a+', 'w+')[epoch==1])
    f.write(str(fid[-1]) +  '\n')
    f.close()

    # graph.set_xdata(np.linspace(0, len(fid), len(fid)))
    # print(np.linspace(0, len(fid)-1, len(fid)))
    # print(fid)
    pad_fid = np.append(fid, np.zeros(model.epochs - len(fid)))
    graph.set_ydata(pad_fid)
    plt.ylim(0, np.max(fid))
    fig.canvas.draw()
    plt.pause(0.001)

    ## DEBUG ##
    print("total_vox:", total_vox, " time_taken:", str(int((time.time() - t0)*10)/10)+'s')
    times += [time.time() - t0]
    h = (model.epochs+1 - epoch) * sum(times) / (60*60*epoch)
    print("Remaining: ~", int(h), "hours", int((h - int(h)) * 60), "minutes")

    show_voxels(gen_vox, save=epoch)

    if epoch % 10 == 0 or epoch == 1:
      torch.save(d_net.state_dict(), 'results/img_res/checkModel.pth')
      print("      Model saved to 'results/img_res/checkModel.pth'")

  # show_images(gen_images, nmax=256)
  torch.save(d_net.state_dict(), 'results/img_res/savedModel.pth')
  print("   Model saved to 'results/img_res/savedModel.pth'")

  print("Minimum FID was", str(np.min(fid)), "at epoch", np.argmin(fid)+1)
  f = open('results/img_res/fid_score.txt', 'a+')
  f.write("Minimum FID was " + str(np.min(fid)) + " at epoch " + str(np.argmin(fid)+1) +  '\n')
  f.close()

  plt.savefig('results/img_res/fid_graph.png')
  plt.ioff()

  # Save data, time and model
  f = open('results/img_res/time.txt', 'w+')
  f.write('avg: ' + str(sum(times)/len(times)) + '\n')
  f.write('total: ' + str(sum(times)) + '\n')
  f.write('\n'.join([str(i) for i in times]))
  f.close()
  l = len(os.listdir('results/'))
  shutil.move('results/img_res', 'results/img_res_' + str(l-2))
  
  


if __name__ == '__main__':
    main()