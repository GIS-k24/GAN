"""
使用Least Squares GAN通过最小二乘法代替二分类的loss

环境：
      torch           1.3.0+cpu
      torchvision     0.4.1+cpu
"""

import torch
from torch import nn
from torch.autograd import Variable

import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_images(images):  # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128


def discriminator():
    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
    )
    return net


def generator(noise_dim=NOISE_DIM):
    net = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return net


def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5


def deprocess_img(x):
    return (x + 1.0) / 2.0


class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


train_set = MNIST(
    root="D:\Python_code\python\PyTorch\Main\img",
    train=True,
    transform=preprocess_img
)
train_data = DataLoader(
    train_set,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_TRAIN, 0)
)


def ls_discriminator_loss(scores_real, scores_fake):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss


def ls_gengrator_loss(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss


def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer


def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, ls_discriminator_loss, ls_gengrator_loss, show_every=250,
                noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = Variable(x).view(bs, -1)  # 真实数据
            logits_real = D_net(real_data)  # 判别网络得分
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5  # -1 ~ 1 的均匀分布

            g_fake_seed = Variable(sample_noise)
            fake_images = G_net(g_fake_seed)  # 生成假数据
            logits_fake = D_net(fake_images)  # 判别网络得分
            d_total_error = ls_discriminator_loss(logits_real, logits_fake)  # 判别器的loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 生成网络
            g_fake_seed = Variable(sample_noise)
            fake_images = G_net(g_fake_seed)  # 生成假数据
            gen_logits_fake = D_net(fake_images)

            g_error = ls_gengrator_loss(gen_logits_fake)  # 生成网络的loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()  # 生成网络优化

            print("------------------------------------")

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1


D = discriminator()
G = generator()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, ls_discriminator_loss, ls_gengrator_loss)
