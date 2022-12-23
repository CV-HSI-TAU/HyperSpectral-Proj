from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from Xception import xception
from model import HyperNet
import matplotlib as cm
import matplotlib.pyplot as plt
from Dataset import AllDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from output_visualization import tensor_to_rgb
from loss_utils import *

# Params
width = 500
height = 400
hyperspectral_dim = 3
input_channels = 350


#Weights defined as rmse , rgb , spectral, L1
weights = [0.0, 0.0, 0.0, 1.0]

# Net and device initialization
# net = HyperNet(n_channels=input_channels, n_output=hyperspectral_dim) # Note hypernet is in Model file
net = xception()
#net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    net = nn.DataParallel(net)
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(), "GPUs Available")
    net = nn.DataParallel(net.cuda())


# Model to test
# main_folder = r'/data/students/adirido/'
main_folder = r'/mnt/storage/datasets/khen_projects/adirido/'
model_name_to_test = r'train_3WL' # Under hypernet2
# net.load_state_dict(torch.load(os.path.join('/home/adirido/PycharmProjects/hyperNet2/', model_name_to_test)))
net.load_state_dict(torch.load(os.path.join('/home/khen_proj_1/hyperNet2/', model_name_to_test)))
path_mono = 'anothertest'            # Folder name under hypernet2
path_hs = 'anothertest'              # Folder name under hypernet2

# Initialization
data_set = AllDataset(height=height, width=width,mono_path=os.path.join(main_folder, path_mono, 'Data Mono'), hs_path=os.path.join(main_folder, path_hs, 'Data HS'))
test = DataLoader(dataset=data_set, batch_size=1)

(mono, hs_tensor) = data_set.__getitem__(0)
# Test loop
# sched = True
# str_w = ' '.join([str(item) for item in weights])
# loss_file = open(str_w + " " + 'scheduler = ' + str(sched) + " testoverfit3" + ".txt", "w")
# loss_file.write(str_w + " " + 'scheduler = ' + str(sched) + '\n')
# epoch =1
# epoch_loss = 0.0


with torch.no_grad():
    for i, (data, labels) in enumerate(test):
        # new_data = torch.zeros(1,input_channels,height,width)
        new_data = torch.zeros_like(data)
        for j in range(input_channels):
          new_data[:,j,:,:] = data[:, 0, :, :] - data[:, j, :, :]
          # sobel1 = sobel(data[:,j,:,:].unsqueeze(0))
          # new_data[:,j,:,:] = sobel(data[:,j,:,:].unsqueeze(0),j)
          # new_data[:,j,:,:] = sobel(data[:, 0, :, :].unsqueeze(0)) -sobel1
        inputs = new_data.to(device)
        labels = labels.to(device)
        output = net(inputs)
        # loss = calcLoss(output, labels, weights, device)
        # loss.backward()
        # epoch_loss += loss.item()
        # loss_file.write('Epoch %s Loss= %s \n' % (epoch + 1, epoch_loss / (i + 1)))
        # epoch += 1

        # fig = plt.figure()
        # zero = torch.zeros_like(data)
        # temp1 = hs_tensor - output[:,:,:,1:501]
        # temp2 = hs_tensor - zero
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(temp1.squeeze(0))
        # plt.axis('off')
        # plt.title("Original minus the output of the net")
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(temp2.squeeze(0))
        # plt.axis('off')
        # plt.title("Original minus zero matrix")
        # plt.show()

        tensor_to_rgb(hs_tensor.unsqueeze(0),output,True,device)

