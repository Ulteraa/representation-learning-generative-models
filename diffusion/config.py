import os.path

import torch
epochs = 500
batch_size = 12
image_size = 64
root_ ='archive/celeba_hq/train'
device = "cuda"
lr = 3e-4
image_chanel=3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
