import torch
import torch.nn as nn

import torch.optim as optim
import enhanced_model_gen, enhanced_model_dis, Perceptual_loss, dataset, config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.utils import save_image
def train_fn(gen, dis, gen_optim, dis_optim, bce, l1, per_loss):
    train_dataset = dataset.Dataset_(config.ROOT)
    train_loader = DataLoader(train_dataset, batch_size=config.Batch_SIZE, shuffle=True)
    loop = tqdm(train_loader, leave=True)
    gen_loss_total = 0
    dis_loss_total = 0
    for _, (low_img, high_img) in enumerate(loop):
        low_img = low_img.to(config.Device)
        high_img = high_img.to(config.Device)
        gen_fake = gen(low_img)
        real = dis(high_img)
        fake = dis(gen_fake.detach())
        # discriminator loss+training
        dis_loss = bce(real, torch.ones_like(real))+bce(fake, torch.zeros_like(fake))
        dis_loss_total += dis_loss
        dis_optim.zero_grad()
        dis_loss.backward()
        dis_optim.step()
        # generator loss+training
        fake = dis(gen_fake)
        gen_loss = bce(fake, torch.ones_like(fake))+per_loss(gen_fake,high_img)+(1e-6)*l1(gen_fake,high_img)
        gen_loss_total += gen_loss
        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()
    print(f' the dis loss is {dis_loss_total:4f} and the gen loss is {gen_loss_total:4f}')

def main():
    test_dataset = dataset.Dataset_(root='set5/test_for_me')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    gen = enhanced_model_gen.Generator(config.IMAG_CHANEL, out_c=64, num_RRDB=23).to(config.Device)
    dis = enhanced_model_dis.Discriminator(config.IMAG_CHANEL).to(config.Device)
    gen_optim=optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    dis_optim=optim.Adam(dis.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    bce=torch.nn.BCEWithLogitsLoss()
    l1=torch.nn.L1Loss()
    per_loss=Perceptual_loss.perceptual_loss()
    for epoch in range(config.NUM_EPOCH):
        train_fn(gen, dis, gen_optim, dis_optim, bce, l1, per_loss)
        if epoch % 10 == 0:
            test(gen, test_loader)
def test(gen, test_loader):
    for _,(low_img, high_img) in enumerate(test_loader):
        with torch.no_grad():
            low_img = low_img.to(config.Device)
            gen_high = gen(low_img)
            path = 'SR_Results/' + 'image_' + str(_) + '.png'
            save_image(gen_high, path)
            path = 'SR_Results/' + 'image_real' + str(_) + '.png'
            save_image(high_img, path)


if __name__=='__main__':
    main()


