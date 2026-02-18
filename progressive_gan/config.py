import torch
import torch.nn as nn
from math import log2
ROOT='archive/celeba_hq/train'
LEARNING_RATE = 1e-3
START_IMG_SIZE = 4 # I am going to start creating images from this resolution and then progress up to 1024
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 16, 8, 4]
IMAGE_CHANEL = 3
LOAD_MODEL = False
SAVE_MODEL = False
Z_dim = 256
IMAGE_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA_GP = 10
NUMS_STEP = int(log2(IMAGE_SIZE/4))+1
PROGRESSIVE_EPOCH = [10]*NUMS_STEP
FIXED_NOISED = torch.randn(16, Z_dim, 1, 1).to(DEVICE)
CRITIC_ITERATION = 1
