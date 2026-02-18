import torch
import torch.nn as nn
from math import log2
from torchvision import transforms
ROOT = 'set5\SR_training_datasets'
from torchvision import transforms

LEARNING_RATE = 1e-4

Batch_SIZE = 64
IMAG_CHANEL = 3
Device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCH = 300
HIGH_RES = 96
LOW_RES = HIGH_RES//4
high_transform=transforms.Compose([transforms.Resize((HIGH_RES, HIGH_RES), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
low_transform=transforms.Compose([transforms.Resize((LOW_RES, LOW_RES), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])])






