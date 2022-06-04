from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

# Params
width = 500
height = 600
hyperspectral_dim = 50
input_channels = 26

# Net and device initialization
net = HyperNet(n_channels=input_channels, n_output=hyperspectral_dim) # Note hypernet is in Model file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
   print("Let's use", torch.cuda.device_count(), "GPUs!")
   net = nn.DataParallel(net)

# Model to test
main_folder = r'/home/adirido/PycharmProjects/hyperNet2/'
model_name_to_test = r' ' # Under hypernet2
net.to(device)
net.load_state_dict(torch.load(os.path.join(main_folder, model_name_to_test)))
path_mono = ''            # Folder name under hypernet2
path_hs = ''              # Folder name under hypernet2

# Initialization
data_set = AllDataset(height=height, width=width,mono_path=os.path.join(main_folder, path_mono, 'Data Mono'), hs_path=os.path.join(main_folder, path_HS, 'Data HS'))
test = DataLoader(dataset=data_set, batch_size=1)

# Test loop
with torch.no_grad():
    for i, (data, labels) in enumerate(test):
        output = net(data)
        tensor_to_rgb(hs_tensor.unsqueeze(0),output,True,device)

