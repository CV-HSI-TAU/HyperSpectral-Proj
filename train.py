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
    hyperspectral_dim = 50
    input_channels = 26


    transforms = trans.Compose([
        trans.RandomVerticalFlip(0.25),
        trans.RandomHorizontalFlip(0.25)
    ])

    ## Net, Dataloader and device:
    net = HyperNet(n_channels = input_channels, n_output = hyperspectral_dim)
    path_mono = ''  # Folder name under hypernet2
    path_hs = ''  # Folder name under hypernet2

    # Initialization
    data_set = AllDataset(height=height, width=width, mono_path=os.path.join(main_folder, path_mono, 'Data Mono'),
                          hs_path=os.path.join(main_folder, path_HS, 'Data HS'))
    trainloader = DataLoader(dataset=data_set, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[0,1,2,3])  
    net.to(device)


    ## Define optimizer and train
    lr = 0.001
    step_size = 200
    gamma = 0.5
    epochs =  1000
    PATH = 'overfit_model'

    ## Run params
    sched = True
    """
    Weights defined as rmse , rgb , spectral, L1
    """
    weights = [0.0, 0.02, 0.0, 1.0]

    ## Loss handlers
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if sched:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    total_loss = []
    start_time=time.time()

    str_w = ' '.join([str(item) for item in weights])
    loss_file = open(str_w + " " + 'scheduler = ' + str(sched) + " overfit" + ".txt", "w")
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
            loss = calcLoss(output, labels, weights, device)
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

        if (epoch+1) % int(epochs/5) == 0:      # Saves checkpoint 5 times in a train
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