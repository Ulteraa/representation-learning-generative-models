import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.utils import save_image
import Pix2Pix_dataset_setup
import Pix2Pix_Discriminator
import PiX2Pix_Generator
import Config
from tqdm import tqdm
def train():
    device=Config.device
    gen=PiX2Pix_Generator.Generator(Config.image_chanel, Config.chanel_dim_gen).to(device)
    dis=Pix2Pix_Discriminator.Discriminator(Config.image_chanel,Config.chanel_dim_dis).to(device)
    PiX2Pix_Generator.intialize_weights(gen)
    Pix2Pix_Discriminator.intialize_weights(dis)
    gen_optimizer=optim.Adam(gen.parameters(),lr=Config.lr_,betas=(0.5,.999))
    dis_optimizer = optim.Adam(dis.parameters(), lr=Config.lr_, betas=(0.5, .999))
    ############################################## training_set
    dataset=Pix2Pix_dataset_setup.pix2pix_dataset(Config.root_training)
    data_loader=DataLoader(dataset=dataset,shuffle=True,batch_size=Config.batch_size)
    ############################################### validation_set
    dataset = Pix2Pix_dataset_setup.pix2pix_dataset(Config.root_validation)
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)

    Criterion=nn.BCEWithLogitsLoss()
    Criterion_rec=nn.L1Loss()

    for epoch in range(Config.epochs):
        loop = tqdm(data_loader, leave=True)
        for _, (image_,target) in enumerate(loop):
            image_=image_.to(device);target=target.to(device)
            y = gen(image_)
            x=dis(image_,target)

            x_fake=dis(image_,y.detach())
            loss=Criterion(x,torch.ones_like(x))+Criterion(x_fake,torch.zeros_like(x_fake))
            dis_optimizer.zero_grad()
            #retain_graph=True
            loss.backward()
            dis_optimizer.step()
            ##########################################
            x_fake = dis(image_, y)
            loss_= Config.lambda_ * Criterion_rec(target, y)+Criterion(x_fake,torch.ones_like(x_fake))
            gen_optimizer.zero_grad()
            loss_.backward()
            gen_optimizer.step()
        print(f'in epoch {epoch}, the dis loss is {loss:4f} and the gen loss is {loss_:4f}')
        with torch.no_grad():
            fake_image_ = gen(image_)
            fake_image_grid = torchvision.utils.make_grid(fake_image_, normalize=True)
            path = 'Pix2Pix_Results/' + 'image_' + str(epoch) + '.png'
            save_image(fake_image_, path)

if __name__=='__main__':
    train()




