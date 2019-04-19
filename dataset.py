import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])
df = pd.read_csv(os.path.join(os.getcwd(), 'train_master.tsv'), delimiter = '\t')
train, valid = train_test_split(df, test_size = 0.2, random_state = 20)

class Trainset(Dataset): 
    def __init__(self, transform = transform, training = True):
        super(Trainset, self).__init__() 
        self.transform = transform
        if training:                            
            self.df = train
        else:
            self.df = valid

    def __getitem__(self, idx):
        path = os.path.join(os.getcwd(), 'train')
        image_path = self.df.iloc[idx][0]
        image_path = os.path.join(path, image_path)
        image_label = self.df.iloc[idx][1]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, image_label

    def __len__(self):
        return len(self.df)