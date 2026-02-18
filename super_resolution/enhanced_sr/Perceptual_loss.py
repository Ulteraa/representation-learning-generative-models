import torch
import torch.nn as nn
from torchvision.models import vgg19
import config

class perceptual_loss(nn.Module):
    def __init__(self):
        super(perceptual_loss, self).__init__()
        self.vgg=vgg19(pretrained=True).eval().to(config.Device)
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self, upscaled_, ground_):
        upscaled_ = self.vgg(upscaled_)
        ground_ = self.vgg(ground_)
        l1 = self.loss(upscaled_, ground_)
        return l1
def test():
    z1 = torch.randn(4, 3, 256, 256)
    z2 = torch.randn(4, 3, 256, 256)
    loss_ = perceptual_loss()
    x=loss_(z1, z2)
    print(x)

if __name__=='__main__':
    test()


