import  torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self,in_chenel, feature=[64,128,256,512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Conv2d(2*in_chenel, feature[0], kernel_size=4, stride=2, padding=1,padding_mode='reflect')
        self.relu = nn.LeakyReLU(0.2)
        in_chenel=feature[0]
        layer=[]
        for feature_ in feature[1:]:
            layer.append(Block_(in_chenel, 2*in_chenel,stride=1 if feature_==feature[-1] else 2))
            in_chenel=2*in_chenel
        layer.append(nn.Conv2d(in_chenel,1,kernel_size=4,padding=1,stride=1,padding_mode='reflect'))
        self.conv=nn.Sequential(*layer)

    def forward(self,x,y):
        x=torch.cat([x,y],dim=1)
        x=self.initial(x)
        x=self.conv(x)
        return x

class Block_(nn.Module):
    def __init__(self, in_chenel, out_chenel,stride):
        super(Block_, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_chenel,out_chenel,kernel_size=4,stride=stride,padding=1,padding_mode='reflect'),
                      nn.BatchNorm2d(out_chenel),
                      nn.LeakyReLU(0.2))
    def forward(self,x):
        return self.conv(x)

def intialize_weights(model):
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal(module.weight.data, 0.0, 0.02)
# def test():
#     in_chenel=3
#     model=Discriminator(in_chenel)
#     intialize_weights(model)
#     x=torch.randn((10,3,256 , 256))
#     y = torch.randn((10,3,256 , 256))
#     predict=model(x,y)
#     return predict.shape
# if __name__=='__main__':
#     print(test())




