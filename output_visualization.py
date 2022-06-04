import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
global wave_lengths

# wl list, taken from imec camera log
wave_lengths= [470, 478.6577181, 487.3154362, 495.9731544, 504.6308725, 513.2885906, 521.9463087, 530.6040268, 539.261745, 547.9194631, 556.5771812, 565.2348993, 573.8926174, 582.5503356, 591.2080537, 599.8657718, 608.5234899, 617.1812081, 625.8389262, 634.4966443, 643.1543624, 651.8120805, 660.4697987, 669.1275168, 677.7852349, 686.442953, 695.1006711, 703.7583893, 712.4161074, 721.0738255, 729.7315436, 738.3892617, 747.0469799, 755.704698, 764.3624161, 773.0201342, 781.6778523, 790.3355705, 798.9932886, 807.6510067, 816.3087248, 824.966443, 833.6241611, 842.2818792, 850.9395973, 859.5973154, 868.2550336, 876.9127517, 885.5704698, 894.2281879]


fig = plt.figure()
rows = 1
columns = 2
def tensor_to_rgb(original, output, isPlot,device):
    RGB_hs=(HStorgb(original,device))
    RGB_NN=(HStorgb(output,device))
    #RGB_NN_new = (RGB_NN-np.nanmin(RGB_NN))/(np.nanmax(RGB_NN)-np.nanmin(RGB_NN))
    if isPlot:
        RGB_NN=RGB_NN.cpu().detach().numpy().astype('float64')
        RGB_hs=RGB_hs.cpu().detach().numpy().astype('float64')
        RGB_NN=np.moveaxis(RGB_NN, 1, 3)
        RGB_hs=np.moveaxis(RGB_hs, 1, 3)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(RGB_hs.squeeze(0))
        plt.axis('off')
        plt.title("Original")
        fig.add_subplot(rows, columns, 2)
        plt.imshow(RGB_NN.squeeze(0))
        plt.axis('off')
        plt.title("Output")
        plt.show()
    else:
        return (RGB_hs), (RGB_NN)

def HStorgb(cube, device):
    weights_r = torch.tensor([0.4335, 1.0026, 0.4479]).to(device)
    weights_g = torch.tensor([0.323,  0.862, 0.995]).to(device)
    weights_b = torch.tensor([1.2876, 0.272, 0.0422]).to(device)
    index_red = wave_lengths.index(677.7852349)
    index_green = wave_lengths.index(547.9194631)
    index_blue = wave_lengths.index(470)
    cube = torch.permute(cube, (0, 2, 3, 1))
    cube.to(device)
    R = torch.matmul(cube[:, :, :, index_red - 1:index_red + 2].to(device), weights_r).unsqueeze(3)
    G = torch.matmul(cube[:, :, :, index_green - 1:index_green + 2].to(device), weights_g).unsqueeze(3)
    B = torch.matmul(cube[:, :, :, index_blue:index_blue + 3].to(device), weights_b).unsqueeze(3)
    R, G, B = map(lambda p: p / (torch.amax(p[0, :, :, :], dim=(0, 1))), [R, G, B])
    R, G, B = map(lambda p: torch.clamp(p, min=0), [R, G, B])
    R = torch.permute(R, (0, 3, 1, 2))
    G = torch.permute(G, (0, 3, 1, 2))
    B = torch.permute(B, (0, 3, 1, 2))
    out = torch.cat((R, G, B), dim=1)
    return out


#Khen code
    #weights_r = torch.tensor([0.4335, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8545, 0.6424, 0.4479]).to(device)
    #weights_g = torch.tensor([0.323, 0.503, 0.71, 0.862, 0.954, 0.995, 0.995, 0.952, 0.87, 0.757]).to(device)
    #weights_b = torch.tensor([1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.272, 0.1582, 0.0783, 0.0422]).to(device)
    #index_red = wave_lengths.index(680.6711409)
    #index_green = wave_lengths.index(550.8053691)
    #index_blue = wave_lengths.index(470)
    #R = cube[:,:,index_red].unsqueeze(0)
    #G = cube[:,:,index_green].unsqueeze(0)
    #B = cube[:,:,index_blue].unsqueeze(0)
    #out = torch.cat([R, G, B], dim=0)
    #return out
    
    #Ido code
    #type = cube.dtype
    #HS = HS.numpy().transpose(1,2,0)
    #bs = HS.shape[0]
    #X = np.dot(HS[ :, :, index_red-5:index_red+5] , weights_r)
    #Y = np.dot(HS[ :, :, index_green-5:index_green+5] , weights_g)
    #Z = np.dot(HS[ :, :, index_blue:index_blue+10] , weights_b)
    #X, Y, Z = map(lambda p: torch.from_numpy(p).type(type), [X, Y, Z])
    #HS = HS.transpose(1,2,0)
    # HS = torch.moveaxis(cube,0,2)
    # bs = HS.shape[0]
    # X = HS[ :, :, index_red-5:index_red+5] @ weights_r
    # Y = HS[ :, :, index_green-5:index_green+5] @ weights_g
    # Z = HS[ :, :, index_blue:index_blue+10] @ weights_b
    # max_val = torch.cat([X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)], dim=1).view(bs, -1).max(-1)[0]
    # X, Y, Z = map(lambda p: p / max_val[:, None, None], [X, Y, Z])
    # X, Y, Z = map(lambda p: torch.clamp(p, min=0), [X, Y, Z])
    # R = 1.25 * X - 0.5 * Y - 0.0 * Z
    # G = - 0.5 * X + 1.6 * Y + 0.0 * Z
    # B = 0.0 * X - 0.2 * Y + 1.7 * Z
    # R, G, B = map(lambda p: torch.clamp(p, min=0, max=1).unsqueeze(1), [R, G, B])
    # out = torch.cat([R, G, B], dim=1)[0, :, :, :]
    # out = torch.permute(out, (1,2,0))
    # return out
def normalize (data):
    data_min = torch.min(input=data, dim=(1, 2), keepdim=True, output=data_min)
    data_max = torch.max(input=data, dim=(1, 2), keepdim=True, output=data_max)
    scaled_data = (data - data_min) / (data_max - data_min)
    return scaled_data