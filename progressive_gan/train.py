import torch
import torch.nn as nn
import Model, config
from math import log2
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
from  tqdm import  tqdm
import torch.optim as optim
def gradient_penalty(cretic, real, fake, alpha, step, device):
    cretic=cretic.to(device)
    N,C,H,W = real.shape
    epsilon_ = torch.randn((N,1,1,1)).repeat(1,C,H,W).to(device)
    x=epsilon_*real+(1-epsilon_)*fake
    mixed_output=cretic(x, alpha, step)
    gradient = torch.autograd.grad(outputs=mixed_output,inputs=x,grad_outputs=torch.ones_like(mixed_output),
                                 retain_graph=True,create_graph=True)[0]
    gradient=gradient.reshape(gradient.shape[0],-1)
    gradient_p=gradient.norm(2, dim=1)
    gradient_p_=torch.mean((gradient_p-1)**2)
    return  gradient_p_

def data_loder(step):
    img_size = 4*(2**step)
    batch_size = config.BATCH_SIZE[step]
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(config.IMAGE_CHANEL)],
                                [0.5 for _ in range(config.IMAGE_CHANEL)])])
    dataset = datasets.ImageFolder(root=config.ROOT,transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE[step], shuffle=True)
    return data_loader, dataset

def train_fn(alpha, step,critic, generator, critic_opt, gen_opt):
    data_loader, dataset=data_loder(step)
    loop = tqdm(data_loader)
    img_num=0
    for _, (img, target) in enumerate(loop):
        img = img.to(config.DEVICE)
        target = target.to(config.DEVICE)
        cur_batch=img.shape[0]
        for n in range(config.CRITIC_ITERATION+1):
            z_noise = torch.randn(cur_batch, config.Z_dim, 1, 1).to(config.DEVICE)
            fake_img = generator(z_noise, alpha, step)
            real = critic(img, alpha, step)
            predict_fake = critic(fake_img, alpha, step)
            gp = gradient_penalty(critic, img, fake_img, alpha, step, config.DEVICE)
            loss = (-(torch.mean(real) - torch.mean(predict_fake)) +
                   config.LAMBDA_GP*gp+0.001*torch.mean(real**2))
            critic_opt.zero_grad()
            loss.backward(retain_graph=True)
            critic_opt.step()
        predict_fake = critic(fake_img, alpha, step)
        loss_g = -torch.mean(predict_fake)
        gen_opt.zero_grad()
        loss_g.backward()
        gen_opt.step()
        alpha += cur_batch/(len(dataset)*config.PROGRESSIVE_EPOCH[step]*0.5)
        alpha = min(alpha, 1)
        if _ % 10 == 0:
            with torch.no_grad():
                fake_fix=generator(config.FIXED_NOISED, alpha, step)
                fake_image_grid = torchvision.utils.make_grid(fake_fix, normalize=True)
                path = 'ProGAN/' + 'image_' + str(img_num) + '.png'
                save_image(fake_image_grid, path)
                img_num += 1


def main():
    critic = Model.Discriminator(config.IMAGE_CHANEL).to(config.DEVICE)
    generator = Model.Generator(config.Z_dim, config.IMAGE_CHANEL).to(config.DEVICE)
    critic_opt = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    gen_opt = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    step = int(log2(config.START_IMG_SIZE/4))
    for num_epoch in (config.PROGRESSIVE_EPOCH[step:]):
        alpha = 1e-8
        print(f'Image size is {4*(2**step)} x {4*(2**step)}')
        for epoch in range(num_epoch):
            train_fn(alpha, step, critic, generator, critic_opt, gen_opt)
        # if save model is True you can start saving your model check points
        step += 1
if __name__=='__main__':
    main()
