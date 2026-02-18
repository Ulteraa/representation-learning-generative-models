import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from model import UNet
import config
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from train import Diffusion
from PIL import Image
import torchvision.transforms as T


def test():
    device = config.device
    model = UNet().to(device)
    PATH = '56ckpt.pt' #24
    model.load_state_dict(torch.load(PATH))
    diffusion = Diffusion(img_size=config.image_size, device=device)
    for i in range(100):
        sampled_images = diffusion.sample(model, n=1)
        sampled_images = sampled_images.squeeze(0)
        sampled_images = (sampled_images .clamp(-1, 1) + 1) / 2
        sampled_images = (sampled_images * 255).type(torch.uint8)
        transform = T.ToPILImage()
        img = transform(sampled_images)
        path ='DIFF_RES/' + str(i)+'.png'
        img = img.save(path)


if __name__ =='__main__':
    test()
