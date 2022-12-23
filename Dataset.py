import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from output_visualization import tensor_to_rgb
from output_visualization import HStorgb

class AllDataset(Dataset):
    def __init__(self, width, height, mono_path, hs_path, transform=None):
        self.folder_HS = hs_path
        self.folder_mono = mono_path
        self.transfrom = transform
        self.mono_sessions = [1,2,3,4]
        self.width = width
        self.height = height
        self.mono_path_list = []
        self.hs_path_list=[]
        data_in_file = sorted(os.listdir(mono_path), key=lambda x: os.path.splitext(x)[0])  # lists all the file that are in that folder
        for i in data_in_file:
            self.mono_path_list += [os.path.join(mono_path, i)]
            self.hs_path_list += [os.path.join(hs_path, i)]
        # for (dirpath, dirnames, filenames) in os.walk(self.folder_mono):  # Recursive search
        #     for dirname in dirnames:
        #         self.mono_path_list += [os.path.join(dirpath, dirname)]
        # for (dirpath, dirnames, filenames) in os.walk(self.folder_HS):  # Recursive search
        #     for dirname in dirnames:
        #         self.hs_path_list += [os.path.join(dirpath, dirname)]

    def __len__(self):
        return len(self.mono_path_list)

    def __getitem__(self, index):
        mono_img_path = self.mono_path_list[index]
        hs_img_path = self.hs_path_list[index]
        mono_images = sorted(os.listdir(mono_img_path), key=lambda x: os.path.splitext(x)[0])  # lists all the file that are in that folder
        hs_images = sorted(os.listdir(hs_img_path), key=lambda x: int(os.path.splitext(x)[0]))

        mono_mat = np.zeros((int(len(mono_images)/2), self.height, self.width))
        # mono_mat = np.zeros((350, self.height, self.width))
        hs_mat = np.zeros((150, self.height, self.width))
        k = 0
        j = 0
        for i in mono_images:
            if k%2==0 and j<350:
                cur_mono = os.path.join(mono_img_path, i)
                mono_mat[j, :, :] = np.array(Image.open(cur_mono).convert("L"), dtype=np.float32) / 255
                j= j+1
            k = k +1

        # j = 0
        # for i in range(0, len(hs_images)-2, 1):
        #     cur_hs1 = os.path.join(hs_img_path, hs_images[i])
        #     cur_hs2 = os.path.join(hs_img_path, hs_images[i+1])
        #     cur_hs3 = os.path.join(hs_img_path, hs_images[i+2])
        #     hs_mat1= np.array(Image.open(cur_hs1).convert("L"), dtype=np.float32) / 255
        #     hs_mat2 = np.array(Image.open(cur_hs2).convert("L"), dtype=np.float32) / 255
        #     hs_mat3 = np.array(Image.open(cur_hs3).convert("L"), dtype=np.float32) / 255
        #     hs_mat1 = np.expand_dims(hs_mat1, axis = 0)
        #     hs_mat2 = np.expand_dims(hs_mat2, axis = 0)
        #     hs_mat3 = np.expand_dims(hs_mat3, axis = 0)
        #     cur_hs_mat = np.mean([hs_mat1, hs_mat2,hs_mat3],axis=0)
        #     hs_mat[j, :, :] = cur_hs_mat
        #     j = j+1
        j=0
        for i in hs_images:
            cur_hs = os.path.join(hs_img_path, i)
            hs_mat[j, :, :] = np.array(Image.open(cur_hs).convert("L"), dtype=np.float32) / 255
            j = j+1


        # (1, H ,W)  ---> (C, H, W)
        # if we are using sigmoind we need:
        # mono_img[mono_img == 255.0] = 1.0
        # hs_img[hs_img == 255.0] = 1.0

        # normalize(mono_mat)
        # normalize(hs_mat)
        mono_tensor = torch.tensor(mono_mat).float()
        hs_tensor = torch.tensor(hs_mat).float()

        if self.transfrom is not None:
            mono_augmentations = self.transfrom(mono_tensor)
            hs_augmentations = self.transfrom(hs_tensor)
            x= torch.equal(mono_tensor,mono_augmentations)
            hs_tensor = hs_augmentations
            mono_tensor = mono_augmentations
        # plt.imshow(mono_tensor[10, :, :])
        # plt.show()
        # plt.imshow(HStorgb(hs_tensor[:, :, :]))
        # plt.show()
        return mono_tensor, hs_tensor
def normalize (data):
    data_min = np.min(data, axis=(1, 2), keepdims=True)
    data_max = np.max(data, axis=(1, 2), keepdims=True)
    scaled_data = (data - data_min) / (data_max - data_min)
    return scaled_data