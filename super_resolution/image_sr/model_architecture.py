import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_chanel, out_chanel, is_discriminator=True, bn=True, act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.is_act = act
        self.conv = nn.Conv2d(in_chanel, out_chanel, bias=not bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanel) if bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2) if is_discriminator else nn.PReLU(out_chanel)
    def forward(self,x):
        return self.act(self.bn(self.conv(x))) if self.is_act else self.bn(self.conv(x))

class upsampling(nn.Module):
    def __init__(self, in_chanel, upscale=2):
        super(upsampling, self).__init__()
        self.up=nn.Sequential(nn.Conv2d(in_chanel, in_chanel*(upscale**2), kernel_size=3, stride=1, padding=1),
                              nn.PixelShuffle(upscale),
                              nn.PReLU(num_parameters=in_chanel)
                              )
    def forward(self,x):
        return self.up(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_chanel):
        super(ResidualBlock, self).__init__()
        self.res=nn.Sequential(ConvBlock(in_chanel, in_chanel, is_discriminator=False, bn=True, act=True, kernel_size=3, stride=1, padding=1),
                               ConvBlock(in_chanel, in_chanel, is_discriminator=False, bn=True, act=False, kernel_size=3, stride=1, padding=1))
    def forward(self,x):
        out=self.res(x)
        return x+out
class Generator(nn.Module):
    def __init__(self, im_chanel, out_chanel=64, num_res=16):
        super(Generator, self).__init__()
        self.initial=ConvBlock(im_chanel, out_chanel, is_discriminator=False, bn=False, act=True, kernel_size=9, stride=1, padding=4)
        self.resblocks = nn.Sequential(*[ResidualBlock(out_chanel) for _ in range(num_res)])
        self.conv = ConvBlock(out_chanel, out_chanel, is_discriminator=False, bn=True, act=False, kernel_size=3, stride=1, padding=1)
        self.up_sample = nn.Sequential(upsampling(out_chanel, upscale=2), upsampling(out_chanel, upscale=2))
        self.final = nn.Conv2d(out_chanel, im_chanel, kernel_size=9, stride=1, padding=4)
    def forward(self,x):
        out = self.initial(x)
        x = out
        x = self.resblocks(x)
        x = self.conv(x)
        x = out+x
        x = self.up_sample(x)
        x = self.final(x)
        return torch.tanh(x)
class Discriminator(nn.Module):
    def __init__(self, in_chanel, feature=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        layer = []
        for index, feature_ in enumerate(feature):
            if index == 0:
                layer.append(ConvBlock(in_chanel, feature_, is_discriminator=True, bn=False, act=True, stride=1+index % 2, kernel_size=3, padding=1))
            else:
                layer.append(ConvBlock(in_chanel, feature_, is_discriminator=True, bn=True, act=True, stride=1+index % 2, kernel_size=3, padding=1))
            in_chanel = feature_
        self.conv = nn.Sequential(*layer)


        self.classifier = nn.Sequential (nn.AdaptiveAvgPool2d((6, 6)),
                                        nn.Flatten(),
                                        nn.Linear(512*6*6, 1024),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(1024, 1))
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

def test():

    z = torch.randn(4, 3, 200, 200)
    model = Generator(im_chanel=3, out_chanel=64, num_res=16)
    gen=model(z)
    print(gen.shape)
    output_ = Discriminator(in_chanel=3)
    print(output_(gen).shape)
if __name__=='__main__':
    test()










