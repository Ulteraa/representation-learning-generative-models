import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import os
import config
from PIL import Image

class Dataset_(Dataset):
    def __init__(self, root):
        self.data = []
        super(Dataset_, self).__init__()
        self.root = root
        self.files = os.listdir(self.root)

        for lable, name in enumerate(self.files):
            dir_ = os.listdir(os.path.join(self.root, name))
            self.data+=list(zip(dir_, [lable]*len(dir_)))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):

        img_, lable_ = self.data[item]
        path_ = os.path.join(self.root, self.files[lable_])
        img = Image.open(os.path.join(path_, img_)).convert('RGB')
        high_img = config.high_transform(img)
        low_img = config.low_transform(img)
        return low_img, high_img
def test():
    DAT=Dataset_(config.ROOT)
    DLOADER=DataLoader(DAT, batch_size=config.Batch_SIZE,shuffle=True)
    for _, (im, tar) in enumerate(DLOADER):
        print(im.shape)
        print(tar.shape)
if __name__=='__main__':
    test()






