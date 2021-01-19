"""
训练GAN
在训练过程中，我们先训练判别网络，让它有一个基本的判断，然后再训练生成网络，然后不断循环
初始时输入GAN中都是随机噪音，然后经过不断的训练，噪音不断转为某种样式的图片，然后迭代变更，生成质量较好的图片

DCGAN
CNN + GAN

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

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置画图的尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


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


def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5


def deprocess_img(x):
    return (x + 1.0) / 2.0


class ChunkSampler(sampler.Sampler):  # 定义一个取样函数
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

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

val_set = MNIST(
    root="D:\Python_code\python\PyTorch\Main\img",
    train=True,
    transform=preprocess_img
)

val_data = DataLoader(
    val_set,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
)

imgs = deprocess_img(train_data.__iter__().next()[0].view(batch_size, 784)).numpy().squeeze()  # 可视化图片
show_images(imgs)

# plt.show()

"""
对抗神经网络：判别网络和生成网络

判别网络的结果是二分类
"""


def discriminator():
    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),  # 判别网络的激活函数
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
    )
    return net


def generator(noise_dim=NOISE_DIM):
    net = torch.nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return net


"""
判别网络的loss和生成网络的loss
"""

bce_loss = nn.BCEWithLogitsLoss()


def discriminator_loss(logits_real, logits_fake):
    size = logits_real.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float()
    false_labels = Variable(torch.zeros(size, 1)).float()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss


def generator_loss(logits_fake):
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float()
    loss = bce_loss(logits_fake, true_labels)
    return loss


"""
参数优化
"""


def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer


# 训练GAN
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, gengrator_loss, show_every=250,
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
            d_total_error = discriminator_loss(logits_real, logits_fake)  # 判别器的loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 生成网络
            g_fake_seed = Variable(sample_noise)
            fake_images = G_net(g_fake_seed)  # 生成假数据
            gen_logits_fake = D_net(fake_images)

            g_error = generator_loss(gen_logits_fake)  # 生成网络的loss
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

train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)
