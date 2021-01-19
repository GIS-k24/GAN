"""
卷积AutoEncoder
img --> code --> img
前者为 NN Encoder，后者为 NN Decoder
NN Encoder：为全连接 + 卷积
NN Decoder：为全连接 + 转置卷积

环境：
      torch           1.3.0+cpu
      torchvision     0.4.1+cpu
"""

import os

import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

# im_tfs = tfs.Compose([
#     tfs.ToTensor(),
#     tfs.Normalize((0.5), (0.5))  # 标准化
# ])

train_set = MNIST(
    root="D:\Python_code\python\PyTorch\Main\img",
    train=True,
    transform=tfs.ToTensor()
)

train_data = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True
)


# 定义网络
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)  # 输出是3维，方便可视化
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


"""
由上述网络也可以看到，使用Linear，输入特征个数为28*28，即输入图片为28*28（如若有自定义的状态，还需进行修改）
"""

net = autoencoder()
x = Variable(torch.randn(1, 28 * 28))  # batch size 是 1
code, _ = net(x)
print(code.shape)

# 然后将图片放入，进行解码和编码
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.MSELoss(size_average=False)


def to_img(x):
    """
    定义一个函数将最后的结果转换回图片    
    :param x: x 
    :return: x
    """
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


for e in range(100):
    for x, y in train_data:
        b_x = Variable(x.view(-1, 28 * 28))
        b_y = Variable(x.view(-1, 28 * 28))
        b_label = Variable(y)

        encoded, decoded = net(b_x)
        loss = criterion(decoded, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) / 20 == 0:
        print(decoded.cpu().data)
