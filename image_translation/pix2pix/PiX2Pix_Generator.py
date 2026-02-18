import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_chanel, out_chanel,down=False,is_dropout=False):
        super(Block, self).__init__()
        if down:
            if is_dropout:
                self.cov=nn.Sequential(nn.Conv2d(in_chanel,out_chanel,kernel_size=4,stride=2, padding=1,padding_mode='reflect'),
                                       nn.BatchNorm2d(out_chanel),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5))
            else:
                self.cov = nn.Sequential(
                    nn.Conv2d(in_chanel, out_chanel, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                    nn.BatchNorm2d(out_chanel),
                    nn.LeakyReLU(0.2))
        else:
            if is_dropout:
                self.cov=nn.Sequential(nn.ConvTranspose2d(in_chanel,out_chanel,kernel_size=4,stride=2, padding=1),
                                       nn.BatchNorm2d(out_chanel),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
            else:
                self.cov = nn.Sequential(
                    nn.ConvTranspose2d(in_chanel, out_chanel, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_chanel),
                    nn.ReLU())
    def forward(self,x):
        return self.cov(x)


class Generator(nn.Module):
    def __init__(self,in_chanel,feature):
        super(Generator, self).__init__()
        self.initial=nn.Sequential(nn.Conv2d(in_chanel,feature,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
                                   nn.LeakyReLU(0.2)) #128x128
        self.d1=Block(feature,2*feature,down=True,is_dropout=False) #64x64
        self.d2=Block(2*feature, 4 * feature, down=True, is_dropout=False) #32x32
        self.d3=Block(feature*4, 8 * feature, down=True, is_dropout=False) #16x16
        self.d4=Block(feature*8, 8* feature, down=True, is_dropout=False) #8x8
        self.d5=Block(feature*8, 8 * feature, down=True, is_dropout=False) #4x4
        self.d6=Block(feature*8, 8 * feature, down=True, is_dropout=False) #2x2
        self.bottleneck=nn.Sequential(nn.Conv2d(feature*8, 8 * feature, kernel_size=4,stride=2,padding=1,padding_mode='reflect'),nn.ReLU()) #1x1

        self.u1 = Block(feature*8, 8*feature, down=False, is_dropout=True)  # 2x2
        self.u2 = Block(feature*2*8, 8 * feature, down=False, is_dropout=True)  # 4x4
        self.u3 = Block(feature*8*2, 8 * feature, down=False, is_dropout=True)  # 8x8
        self.u4 = Block(feature*8*2, 8 * feature, down=False, is_dropout=False)  # 16x16
        self.u5 = Block(feature*8*2, 4* feature, down=False, is_dropout=False)  # 32x32
        self.u6 = Block(feature*4*2, 2 * feature, down=False, is_dropout=False)  # 64x64
        self.u7 = Block(feature *2*2, feature, down=False, is_dropout=False)  # 128x128
        self.final=nn.Sequential(nn.ConvTranspose2d(2*feature, 3,kernel_size=4,stride=2,padding=1),
        nn.Tanh())

    def forward(self,x):
        x=self.initial(x)
        d1=self.d1(x)
        d2 =self.d2(d1)
        d3 =self.d3(d2)
        d4 =self.d4(d3)
        d5 =self.d5(d4)
        d6 =self.d6(d5)
        d7 = self.bottleneck(d6)
        up1 =self.u1(d7)
        up2 =self.u2(torch.cat([d6,up1], dim=1))
        up3 =self.u3(torch.cat([d5,up2], dim=1))
        up4 =self.u4(torch.cat([d4,up3], dim=1))
        up5 =self.u5(torch.cat([d3,up4], dim=1))
        up6 =self.u6(torch.cat([d2,up5], dim=1))
        up7 =self.u7(torch.cat([d1,up6], dim=1))
        x=self.final(torch.cat([x,up7], dim=1))
        return x

def intialize_weights(model):
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal(module.weight.data, 0.0, 0.02)
# def test():
#     in_chenel=3
#     model=Generator(in_chenel,feature=8)
#     intialize_weights(model)
#     x=torch.randn((1,3,256 , 256))
#     predict=model(x)
#     return predict.shape
# if __name__=='__main__':
#     print(test())






