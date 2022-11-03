import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('./dataset/train_img.csv', header=None)
        label = pd.read_csv('./dataset/train_label.csv', header=None)
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('./dataset/test_img.csv', header=None)
        return np.squeeze(img.values), None


class SimpsonsLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""
        # print("name: ", self.img_name[index] + '.jpeg')

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
        """
        img_path = os.path.join(self.root, self.img_name[index])
        img = Image.open(img_path).convert('RGB')

        """
           step2. Get the ground truth label from self.label
        """
        if self.mode == "train":
            label = self.label[index].copy()
        
        """             
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
        """
        # img = img/255
        # print(img.size)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(np.transpose(img, (2,0,1)))

        # print(img.size())

        if self.mode == "train":
            label = torch.tensor(label).long()
        else:
            label = None
        """
            step4. Return processed image and label
        """
        # print(type(img), type(label))
        # return (img.unsqueeze(0), label.unsqueeze(0))
        return (img, label)


if __name__ == '__main__':
    train_data = SimpsonsLoader(".", "train")
    for i in range(100):
        img, label = train_data[i]

    test_data = SimpsonsLoader(".", "test")
    for i in range(100):
        img, label = test_data[i]
