import torch
import torch.nn as nn
class ConVBlock(nn.Module):
    def __init__(self, in_c, out_c, act=True, **kwargs):
        super(ConVBlock, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_c, out_c, **kwargs),
        nn.LeakyReLU(0.2) if act else nn.Identity())

    def forward(self, x):
        return self.conv(x)

class Upsampling(nn.Module):
    def __init__(self, in_c, scale_factor):
        super(Upsampling, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                                nn.Conv2d(in_c, in_c, 3, 1, 1),
                                nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.up(x)
class DenseBlock(nn.Module):
    def __init__(self, in_c, out_c=64, beta=0.2):
        super(DenseBlock, self).__init__()
        self.beta=beta
        self.blocks = nn.ModuleList()
        for i in range(5):
            self.blocks.append(ConVBlock(in_c+i*out_c, out_c, act=True if i <= 3 else False,
                                         kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        input_x = x
        for block in self.blocks:
            out = block(input_x)
            input_x = torch.cat([input_x, out], dim=1)
        return x+self.beta*out
class RRDB(nn.Module):
    def __init__(self, in_c, out_c, beta=0.2):
        super(RRDB, self).__init__()
        self.beta=beta
        self.rrdb = nn.Sequential(*[DenseBlock(in_c, out_c) for _ in range(3)])
    def forward(self, x):
        input_x = x
        out = self.rrdb(x)
        return input_x + out * self.beta
class Generator (nn.Module):
    def __init__(self, img_c, out_c, num_RRDB):
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(img_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.rrdblocks = nn.Sequential(*[RRDB(out_c, out_c) for _ in range(num_RRDB)])
        self.conv = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.up = nn.Sequential(Upsampling(out_c, scale_factor=2), Upsampling(out_c, scale_factor=2))
        self.final=nn.Sequential(nn.Conv2d(out_c, out_c, 3, 1, 1),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(out_c, img_c, 3, 1, 1))
    def forward(self, x):
        out = self.initial(x)
        out_b = out
        out = self.rrdblocks(out)
        out = self.conv(out)
        out = self.up(out+out_b)
        out = self.final(out)
        return out

def test():

    z = torch.randn(4, 3, 24, 24)
    model = Generator(img_c=3, out_c=64, num_RRDB=23)
    gen=model(z)
    print(gen.shape)

if __name__=='__main__':
    test()








        

