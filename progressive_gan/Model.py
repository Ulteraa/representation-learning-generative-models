import torch
import torch.nn as nn
feature = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
class WSconv(nn.Module):
    def __init__(self,in_c, out_c, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSconv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        nn.init.normal_(self.conv.weight)
        self.scale = (gain/(out_c*kernel_size**2))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.zeros_(self.bias)
    def forward(self, x):
        return self.conv(x*self.scale)+self.bias.view(1, self.bias.shape[0], 1, 1)
class Pixel_Norm(nn.Module):
    def __init__(self, done=True):
        super(Pixel_Norm, self).__init__()
        self.done = done
        self.epsilon = 10**(-8)

    def forward(self, x):
        if self.done:
            return x / torch.sqrt(torch.mean(x ** 2, dim=1,keepdim=True) + self.epsilon)
        return x
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c,act_pn=True):
        super(ConvBlock, self).__init__()
        self.act_pn=act_pn
        self.conv=nn.Sequential(WSconv(in_c, out_c,kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.2),
                                Pixel_Norm(act_pn),
                                WSconv(out_c, out_c, kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.2),
                                Pixel_Norm(act_pn),
                                )
    def forward(self,x):
        return self.conv(x)
class Generator(nn.Module):
    def __init__(self,z_dim,im_chanel):
        super(Generator, self).__init__()
        self.feature=[int(512*_) for _ in feature]
        self.initial=nn.Sequential(nn.ConvTranspose2d(z_dim, self.feature[0], kernel_size=4, stride=1, padding=0),
                                   nn.LeakyReLU(0.2),
                                   Pixel_Norm(done=True),
                                   WSconv(self.feature[0], self.feature[0], kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(0.2),
                                   Pixel_Norm(done=True))
        self.init_rgb=WSconv(self.feature[0], im_chanel, kernel_size=1, stride=1, padding=0)
        self.modules_, self.rgb_layes = nn.ModuleList(), nn.ModuleList([self.init_rgb])
        for i in range(len(self.feature)-1):
            self.modules_.append(ConvBlock(self.feature[i], self.feature[i+1], act_pn=True))
            self.rgb_layes.append(WSconv(self.feature[i+1], im_chanel, kernel_size=1, stride=1, padding=0))


    def fed_(self,alpha,upsacaled_img,gen_img):
        return torch.tanh((1-alpha)*upsacaled_img+alpha*gen_img)
    def forward(self,x,alpha,step):
        x = self.initial(x)
        if step == 0:
            x = self.init_rgb(x)
            return x
        for i in range(step):
            upscaled_ = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.modules_[i](upscaled_)
        out = self.fed_(alpha, self.rgb_layes[step-1](upscaled_), self.rgb_layes[step](x))
        return out

class Discriminator(nn.Module):
    def __init__(self,im_chanel):
        super(Discriminator, self).__init__()
        n=len(feature)-1
        feature.reverse()
        self.leaky = nn.LeakyReLU(0.2)
        self.feature = [int(512 * _) for _ in feature]
        self.AvgPool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.intitial = nn.Sequential(WSconv(in_c=1+self.feature[n], out_c=self.feature[n], kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2),
                                      WSconv(in_c=self.feature[n], out_c=self.feature[n], kernel_size=4, stride=1, padding=0),
                                      nn.LeakyReLU(0.2),
                                      WSconv(self.feature[n], 1, kernel_size=1, stride=1,padding=0))
        self.int_rgb_to_ten=WSconv(im_chanel, self.feature[n], kernel_size=1, stride=1, padding=0)
        self.modules_, self.ten_to_rgb_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(n):
            self.modules_.append(ConvBlock(in_c=self.feature[i], out_c=self.feature[i+1], act_pn=False))
            self.ten_to_rgb_layers.append(WSconv(in_c=im_chanel, out_c=self.feature[i], kernel_size=1, stride=1, padding=0))
        self.ten_to_rgb_layers.append(self.int_rgb_to_ten)

    def fed_ (self,img, down_sample, alpha):
        return (1-alpha)*img+ alpha*down_sample

    def forward(self, img, alpha, step):
         if step == 0:
             x = self.leaky(self.int_rgb_to_ten(img))
             x = self.minibatch_stddev(x)
             x = self.intitial(x)
             return x

         d = len(self.feature)-1
         step = d-step
         out = self.leaky(self.ten_to_rgb_layers[step](img))

         out = self.fed_(alpha, self.leaky(self.ten_to_rgb_layers[step+1](self.AvgPool(img))) ,self.AvgPool(self.modules_[step](out)))
         for i in range(step+1, d):
                 out = self.modules_[i](out)
                 out = self.AvgPool(out)

         out= self.minibatch_stddev(out)
         out = self.intitial(out)
         return out

    def minibatch_stddev(self,x):
        y = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, y], dim=1)


def test():
    x=torch.randn(1, 3, 1024, 1024)
    # model=Generator(z_dim=100,im_chanel=3)
    model = Discriminator(im_chanel=3)
    # z=torch.randn(2,100,1,1)
    alpha = 0.5
    for step in range(len(feature)):
        x = torch.randn(2, 3, 4*(2**step), 4*(2**step))
        x = model(x, alpha, step)
        print(x.shape)

if __name__=='__main__':
    test()
