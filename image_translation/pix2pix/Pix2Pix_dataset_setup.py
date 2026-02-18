import torch
import  torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import  transforms
import Config
import  os
from PIL import  Image
import  numpy as np
import tqdm
class pix2pix_dataset(Dataset):
    def __init__(self, root):
        super(pix2pix_dataset, self).__init__()
        self.list_=os.listdir(root)
        self.root=root
    def __len__(self):
        return len(self.list_)
    def __getitem__(self, item):
        path=os.path.join(self.root,self.list_[item])
        image=np.asarray(Image.open(path))
        image_,target_=image[:,0:600,:],image[:,600:,:]
        image_= Image.fromarray(image_)
        target_ = Image.fromarray(target_)
        # augmentation=Config.transform_to_both(image=image_,image0=target_)
        # image_,target_=augmentation['image'],augmentation['image0']
        image_=Config.transform(image_)
        target_ = Config.transform(target_)
        return image_,target_




        

