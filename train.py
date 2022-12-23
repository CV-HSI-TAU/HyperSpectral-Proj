# IMPORTS
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from output_visualization import tensor_to_rgb
from model import HyperNet
from Xception import xception
import matplotlib as cm
import matplotlib.pyplot as plt
from Dataset import AllDataset
from torchvision.transforms import transforms as trans
import torch
import pytorch_ssim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
#import numpy as np
from loss_utils import *
import datetime
import time
from torch.autograd import Variable
import cv2 as cv
import os
#
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
if __name__ == '__main__':

    ## Define parameters:

    width = 500
    height = 400
    hyperspectral_dim = 3
    input_channels = 350


    #net = HyperNet(n_channels = input_channels, n_output = hyperspectral_dim)
    net = xception()
    transforms = trans.Compose([
        trans.RandomVerticalFlip(0.25),
        trans.RandomHorizontalFlip(0.25)
        #trans.RandomCrop,
        #trans.RandomRotation
    ])

    
    #function for edges

    def sobel(image):
        kernel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]])
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        kernel_y = torch.tensor([[-1., -2., -1.],
                             [0., 0., 0.],
                             [1., 2., 1.]])
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

        photo = image
        result_x = F.conv2d(photo, kernel_x,padding=1)
        result_y = F.conv2d(photo, kernel_y,padding=1)
        result = torch.sqrt(torch.mul(result_x, result_x) + torch.mul(result_y, result_y))
        # plt.imsave('/home/adirido/PycharmProjects/hyperNet2/sobel/'+str(q)+'.png' ,result.squeeze(0).squeeze(0))
        return result

        
    ## Dataloader and device:
    # data_set = AllDataset(height=height, width=width,mono_path=r'/data/students/adirido/overfit/Data Mono/', hs_path=r'/data/students/adirido/overfit/Data HS/', transform = transforms)
    data_set = AllDataset(height=height, width=width, mono_path=r'/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data Mono/',
                          hs_path=r'/mnt/storage/datasets/khen_projects/adirido/Data_HS_Mono/Data HS/', transform=transforms)
    trainloader = DataLoader(dataset=data_set, num_workers=12, batch_size=1, shuffle = False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
    net.to(device)
    #if torch.cuda.device_count() > 1:
        #print(torch.cuda.device_count(), "GPUs Available")
        #net = nn.DataParallel(net.cuda())  


    ## Define optimizer and train
    lr = 0.001
    step_size = 200
    gamma = 0.5
    epochs =  2000
    PATH = 'train_3WL'

    ## Run params
    sched = True
    """
    Weights defined as rmse , rgb , spectral, L1
    """
    weights = [0.0, 0.0, 0.0, 1.0]

    ## Loss handlers
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if sched:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    total_loss = []
    start_time=time.time()

    str_w = ' '.join([str(item) for item in weights])
    loss_file = open(str_w + " " + 'scheduler = ' + str(sched) + " train_3WL" + ".txt", "w")
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

            new_data = torch.zeros_like(data)
            # sobel1 = torch.zeros(12,1,height,width)
            for j in range(input_channels):
                new_data[:,j,:,:] = data[:, 0, :, :] - data[:, j, :, :]
                # sobel1 = sobel(data[:,j,:,:].unsqueeze(0))
                # new_data[:,j,:,:] = sobel(data[:,j,:,:].unsqueeze(0),j)
                # new_data[:,j,:,:] = sobel(data[:, 0, :, :].unsqueeze(0)) -sobel1
                # plt.imsave('/home/adirido/PycharmProjects/hyperNet2/sobel/' + str(j) + '.png',new_data[:,j,:,:].squeeze(0).squeeze(0),cmap='gray')



            inputs = new_data.to(device)
            #inputs = new_data.cuda()
            #inputs = data.to(device)
            labels = labels.to(device)
            #labels = labels.cuda()

            
            optimizer.zero_grad()

            output = net(inputs)

            #loss = calcLoss(output.cpu().detach(), labels.cpu().detach(), weights)
            loss = calcLoss(output[:,:,:,1:501], labels, weights,device)
            #loss_var = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

#saving the data,label and output to check the code
        # img1 = labels.cpu().data
        # img2 = output.cpu().data
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(img1.permute(0, 3, 2, 1).squeeze(0)[:, :, 75])
        # plt.axis('off')
        # plt.title("labels")
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(data.permute(0, 3, 2, 1).squeeze(0)[:, :, 75])
        # plt.axis('off')
        # plt.title("input")
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(img2.permute(0, 3, 2, 1).squeeze(0)[:,:,75])
        # plt.axis('off')
        # plt.title("output")
        # plt.show()

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