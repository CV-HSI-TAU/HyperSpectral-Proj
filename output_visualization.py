import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
global wave_lengths
# wave_lengths= [
#     470, 472.885906, 475.7718121, 478.6577181, 481.5436242, 484.4295302, 487.3154362, 490.2013423, 493.0872483,
#     495.9731544, 498.8590604, 501.7449664, 504.6308725, 507.5167785, 510.4026846, 513.2885906, 516.1744966, 519.0604027,
#     521.9463087, 524.8322148, 527.7181208, 530.6040268, 533.4899329, 536.3758389, 539.261745, 542.147651, 545.033557,
#     547.9194631, 550.8053691, 553.6912752, 556.5771812, 559.4630872, 562.3489933, 565.2348993, 568.1208054, 571.0067114,
#     573.8926174, 576.7785235, 579.6644295, 582.5503356, 585.4362416, 588.3221477, 591.2080537, 594.0939597, 596.9798658,
#     599.8657718, 602.7516779, 605.6375839, 608.5234899, 611.409396, 614.295302, 617.1812081, 620.0671141, 622.9530201,
#     625.8389262, 628.7248322, 631.6107383, 634.4966443, 637.3825503, 640.2684564, 643.1543624, 646.0402685, 648.9261745,
#     651.8120805, 654.6979866, 657.5838926, 660.4697987, 663.3557047, 666.2416107, 669.1275168, 672.0134228, 674.8993289,
#     677.7852349, 680.6711409, 683.557047, 686.442953, 689.3288591, 692.2147651, 695.1006711, 697.9865772, 700.8724832,
#     703.7583893, 706.6442953, 709.5302013, 712.4161074, 715.3020134, 718.1879195, 721.0738255, 723.9597315, 726.8456376,
#     729.7315436, 732.6174497, 735.5033557, 738.3892617, 741.2751678, 744.1610738, 747.0469799, 749.9328859, 752.8187919,
#     755.704698, 758.590604, 761.4765101, 764.3624161, 767.2483221, 770.1342282, 773.0201342, 775.9060403, 778.7919463,
#     781.6778523, 784.5637584, 787.4496644, 790.3355705, 793.2214765, 796.1073826, 798.9932886, 801.8791946, 804.7651007,
#     807.6510067, 810.5369128, 813.4228188, 816.3087248, 819.1946309, 822.0805369, 824.966443, 827.852349, 830.738255,
#     833.6241611, 836.5100671, 839.3959732, 842.2818792, 845.1677852, 848.0536913, 850.9395973, 853.8255034, 856.7114094,
#     859.5973154, 862.4832215, 865.3691275, 868.2550336, 871.1409396, 874.0268456, 876.9127517, 879.7986577, 882.6845638,
#     885.5704698, 888.4563758, 891.3422819, 894.2281879, 897.114094, 900
# ]

#wave_lengths= [470, 478.6577181, 487.3154362, 495.9731544, 504.6308725, 513.2885906, 521.9463087, 530.6040268, 539.261745, 547.9194631, 556.5771812, 565.2348993, 573.8926174, 582.5503356, 591.2080537, 599.8657718, 608.5234899, 617.1812081, 625.8389262, 634.4966443, 643.1543624, 651.8120805, 660.4697987, 669.1275168, 677.7852349, 686.442953, 695.1006711, 703.7583893, 712.4161074, 721.0738255, 729.7315436, 738.3892617, 747.0469799, 755.704698, 764.3624161, 773.0201342, 781.6778523, 790.3355705, 798.9932886, 807.6510067, 816.3087248, 824.966443, 833.6241611, 842.2818792, 850.9395973, 859.5973154, 868.2550336, 876.9127517, 885.5704698, 894.2281879]
#wave_lengths=[470,513.2885906 ,556.5771812 ,599.8657718 ,643.1543624 ,686.442953, 729.7315436 ,773.0201342, 816.3087248 ,859.5973154]
wave_lengths=[470,550.8053691,700.8724832]
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

#for 10 WL
#def HStorgb(cube, device):
#    weights_r = torch.tensor([1.0]).to(device)
#    weights_g = torch.tensor([0.5,0.5]).to(device)
#    weights_b = torch.tensor([1.0]).to(device)
#    index_red = wave_lengths.index(686.442953)
#    index_green = wave_lengths.index(556.5771812)
#    index_blue = wave_lengths.index(470)
#    cube = torch.permute(cube, (0, 2, 3, 1))
#    cube.to(device)
#    R = torch.matmul(cube[:, :, :, index_red:index_red+1].to(device), weights_r).unsqueeze(3)
#    G = torch.matmul(cube[:, :, :, index_green - 1:index_green+1].to(device), weights_g).unsqueeze(3)
#    B = torch.matmul(cube[:, :, :, index_blue:index_blue+1].to(device), weights_b).unsqueeze(3)
#    R, G, B = map(lambda p: p / (torch.amax(p[0, :, :, :], dim=(0, 1, 2))), [R, G, B])
#    R, G, B = map(lambda p: torch.clamp(p, min=0), [R, G, B])
#    R = torch.permute(R, (0, 3, 1, 2))
#    G = torch.permute(G, (0, 3, 1, 2))
#    B = torch.permute(B, (0, 3, 1, 2))
#    out = torch.cat((R, G, B), dim=1)
#    return out

#for 3 WL
def HStorgb(cube, device):
    weights_r = torch.tensor([1.0]).to(device)
    weights_g = torch.tensor([1.0]).to(device)
    weights_b = torch.tensor([0.6]).to(device)
    index_red = wave_lengths.index(700.8724832)
    index_green = wave_lengths.index(550.8053691)
    index_blue = wave_lengths.index(470)
    cube = torch.permute(cube, (0, 2, 3, 1))
    cube.to(device)
    R = torch.matmul(cube[:, :, :, index_red:index_red+1].to(device), weights_r).unsqueeze(3)
    G = torch.matmul(cube[:, :, :, index_green:index_green+1].to(device), weights_g).unsqueeze(3)
    B = torch.matmul(cube[:, :, :, index_blue:index_blue+1].to(device), weights_b).unsqueeze(3)
    R, G, B = map(lambda p: p / (torch.amax(p[0, :, :, :], dim=(0, 1, 2))), [R, G, B])
    R, G, B = map(lambda p: torch.clamp(p, min=0), [R, G, B])
    R = torch.permute(R, (0, 3, 1, 2))
    G = torch.permute(G, (0, 3, 1, 2))
    B = torch.permute(B, (0, 3, 1, 2))
    out = torch.cat((R, G, B), dim=1)
    return out
    
#for 50 WL
#def HStorgb(cube, device):
#    weights_r = torch.tensor([1.0, 0.5, 0.05]).to(device)
#    weights_g = torch.tensor([0.5,1,0.5]).to(device)
#    weights_b = torch.tensor([0.5,1.0,0.5]).to(device)
#    index_red = wave_lengths.index(703.7583893)
#    index_green = wave_lengths.index(556.5771812)
#    index_blue = wave_lengths.index(470)
#    cube = torch.permute(cube, (0, 2, 3, 1))
#    cube.to(device)
#    R = torch.matmul(cube[:, :, :, index_red-1:index_red+2].to(device), weights_r).unsqueeze(3)
#    G = torch.matmul(cube[:, :, :, index_green - 1:index_green+2].to(device), weights_g).unsqueeze(3)
#    B = torch.matmul(cube[:, :, :, index_blue:index_blue+3].to(device), weights_b).unsqueeze(3)
#    R, G, B = map(lambda p: p / (torch.amax(p[0, :, :, :], dim=(0, 1, 2))), [R, G, B])
#    R, G, B = map(lambda p: torch.clamp(p, min=0), [R, G, B])
#   R = torch.permute(R, (0, 3, 1, 2))
#    G = torch.permute(G, (0, 3, 1, 2))
# B = torch.permute(B, (0, 3, 1, 2))
#   out = torch.cat((R, G, B), dim=1)
#    return out

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