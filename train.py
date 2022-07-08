# IMPORTS
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from output_visualization import tensor_to_rgb
from model import HyperNet
import matplotlib as cm
import matplotlib.pyplot as plt
from Dataset import AllDataset
from torchvision.transforms import transforms as trans
import torch
import pytorch_ssim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
#import numpy as np
from loss_utils import *
import datetime
import time
from torch.autograd import Variable
#

if __name__ == '__main__':

    ## Define parameters:

    width = 500
    height = 600
    hyperspectral_dim = 3
    input_channels = 26


    net = HyperNet(n_channels = input_channels, n_output = hyperspectral_dim)
    transforms = trans.Compose([
        trans.RandomVerticalFlip(0.25),
        trans.RandomHorizontalFlip(0.25)
    ])
    

    ## Dataloader and device:
    data_set = AllDataset(height=height, width=width,mono_path=r'/data/students/adirido/Data_HS_Mono/Data Mono/', hs_path=r'/data/students/adirido/Data_HS_Mono/Data HS/', transform = transforms)
    trainloader = DataLoader(dataset=data_set, batch_size=6, shuffle = True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[0,1,2,3])  
    net.to(device)


    ## Define optimizer and train
    lr = 0.001
    step_size = 600
    gamma = 0.5
    epochs =  5000
    PATH = 'train_model_with_SSIM_0.4_3WL_with_aug'

    ## Run params
    sched = True
    """
    Weights defined as rmse , rgb , spectral, L1
    """
    weights = [0.0, 0.4, 0.0, 0.8] #Sum 1.2

    ## Loss handlers
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if sched:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    total_loss = []
    start_time=time.time()

    str_w = ' '.join([str(item) for item in weights])
    loss_file = open(str_w + " " + 'scheduler = ' + str(sched) + " train_3WL_with_aug" + ".txt", "w")
    loss_file.write(str_w + " " + 'scheduler = ' + str(sched) + '\n')
    loss_file.write("Started executing at: " + str( start_time ) + '\n')
    print('Start Training The Model')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (data, labels) in enumerate(trainloader):
            """
            # inputs size (N, 28, H, W)
            # labels size (N, hyperspectral_dim, H, W)
            # output size (N, hyperspectral_dim, H, W)            
            """
            inputs = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = net(inputs)
            #loss = calcLoss(output.cpu().detach(), labels.cpu().detach(), weights)
            loss = calcLoss(output, labels, weights,device)
            #loss_var = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()


        print('Epoch', epoch + 1, 'Loss =', epoch_loss/(i+1))
        loss_file.write('Epoch %s Loss= %s \n' % (epoch+1 ,epoch_loss/(i+1)))
        total_loss += [epoch_loss]

        if sched:
            scheduler.step()

        if (epoch+1) % (int(epochs)/5) == 0:
            torch.save(net.state_dict(), PATH)
            print('Saved Model', PATH)

    print('Finished Training The Model')

    torch.save(net.state_dict(), PATH)
    print('Saved Model', PATH)
    end_time=time.time()
    loss_file.write("Finished running at: " + str( start_time ) + '\n' + 'took: ' + str( end_time - start_time))
    loss_file.close()
    #saving last epoch run
    output = output.cpu().detach()
    output_mat = output.numpy().astype('float64')