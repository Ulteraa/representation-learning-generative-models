import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from W_GAN_GP_model import Critic, Generator, initialize_weight, gradient_penalty
from torchvision.utils import save_image

# hyperparameter
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr_ = 1e-4
epochs = 100
image_chenel = 1
feature_dim = 64
z_dim = 100
critic_iteration = 1
lambda_GP = 10
dis_ = Critic(image_chenel, feature_dim).to(device)
gen_ = Generator(z_dim, image_chenel, feature_dim).to(device)
initialize_weight(dis_)
initialize_weight(gen_)
optimizer_critic = optim.Adam(dis_.parameters(), lr=lr_, betas=(0.0, 0.9))
optimizer_gen = optim.Adam(gen_.parameters(), lr=lr_, betas=(0.0, 0.9))
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(image_chenel)],
                                                     [0.5 for _ in range(image_chenel)])])
dataset = torchvision.datasets.MNIST(root='MNIST/', transform=transform, download=True)
data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
fix_noise = torch.randn(40, 100, 1, 1).to(device)
print(device)
for epoch in range(epochs):
    for _, (image, target) in enumerate(data_loader):
        image = image.to(device)
        target = target.to(device)
        for i in range(critic_iteration):
            noise = torch.randn((image.shape[0], z_dim, 1, 1)).to(device)
            fake_image = gen_(noise)
            real_lable = dis_(image).reshape(-1)
            fake_lable = dis_(fake_image).reshape(-1)
            gp = gradient_penalty(dis_, image, fake_image, device)
            loss_dis = -(torch.mean(real_lable) - torch.mean(fake_lable))+lambda_GP*gp
            optimizer_critic.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_critic.step()
        #############
        f = dis_(fake_image).reshape(-1)
        loss_gen = -(torch.mean(f))
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

    print(f'in epoch {epoch}, the dis loss is {loss_dis:4f} and the gen loss is {loss_gen:4f}')
    with torch.no_grad():
        fake_image_ = gen_(fix_noise)
        fake_image_grid = torchvision.utils.make_grid(fake_image_, normalize=True)
        path = 'WGAN_GP_Results/' + 'image_' + str(epoch) + '.png'
        save_image(fake_image_grid, path)
