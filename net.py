import torch
import numpy as np
from torch import nn
from util import cal_m_auc


def create_conv_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
    )


class CNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        stride = 1 if k == 500 else 2 if k == 1000 else 3
        self.conv_block1 = create_conv_block(4, 128, 1, stride)  # shape: [n, 288, 500]
        self.conv_block2 = create_conv_block(128, 512, 4, 4)  # shape: [n, 512, 125]
        self.conv_block3 = create_conv_block(512, 128, 1, 4)  # shape: [n, 128, 32]
        self.conv_block4 = create_conv_block(128, 64, 4, 4)  # shape: [n, 64, 8]
        self.linear_block = nn.Sequential(
            nn.Linear(64 * 8, 10),
            nn.GELU(),
            nn.Linear(10, 2000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.reshape(-1, 64 * 8)
        x = self.linear_block(x)
        return x


def train(net, loader, cri, opt, device, max_epoch):
    loss_list = []
    for epoch in range(max_epoch):
        cur_loss = .0
        for idx, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            output = net(data)
            loss = cri(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            cur_loss += loss.item()
        cur_loss /= len(loader)
        loss_list.append(cur_loss)
        print('epoch: {}, loss: {:.6f}'.format(epoch+1, cur_loss))
    return loss_list


def test(net, loader, label, device):
    net.eval()
    res = []
    for idx, (data, label) in enumerate(loader):
        data = data.to(device)
        output = net(data)
        output = output.detach().cpu().numpy()
        res.append(output)
    res = np.array(res)
    return cal_m_auc(res, label)
