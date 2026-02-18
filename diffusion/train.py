# import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from model import UNet
import config
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=config.image_size, device=config.device):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        x = (x * 255)
        return x

def data_loder():
    img_size = config.image_size
    batch_size = config.batch_size
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(config.image_chanel)],
                                [0.5 for _ in range(config.image_chanel)])])
    dataset = datasets.ImageFolder(root=config.root_, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train():

    device = config.device
    dataloader = data_loder()
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=config.image_size, device=device)

    l = len(dataloader)

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        sampled_images = diffusion.sample(model, n=images.shape[0])
        path = 'Results/' + 'image_' + str(epoch) + '.png'
        save_image(sampled_images, path)
        torch.save(model.state_dict(), f"ckpt.pt")
if __name__=='__main__':
    train()




