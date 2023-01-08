import torch
import numpy as np
from torch import nn


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
        self.conv_block1 = create_conv_block(4, 64, 1, stride)  # shape: [n, 64, 500]
        self.conv_block2 = create_conv_block(64, 128, 7, 4)  # shape: [n, 128, 124]
        self.conv_block3 = create_conv_block(128, 256, 5, 4)  # shape: [n, 256, 30]
        self.conv_block4 = create_conv_block(256, 512, 5, 4)  # shape: [n, 512, 7]
        self.linear_block = nn.Sequential(
            nn.Linear(512 * 7, 64),
            nn.Dropout(.2),
            nn.GELU(),
            nn.Linear(64, 2000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.reshape(-1, 512 * 7)
        x = self.linear_block(x)
        return x


class AlternativeCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        stride = 1 if k == 500 else 2 if k == 1000 else 3
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(4, 64, (1, ), (stride, )),  # shape: [n, 64, 500]
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2, 2),  # shape: [n, 64, 250]
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 256, (3, ), (4, )),  # shape: [n, 256, 62]
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2, 2),  # shape: [n, 256, 31]
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, 512, (5, ), (2, )),  # shape: [n, 512, 14]
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(2, 2)  # shape: [n, 512, 7]
        )
        self.linear_block = nn.Sequential(
            nn.Linear(512 * 7, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(.2),
            nn.GELU(),
            nn.Linear(64, 2000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.reshape(-1, 512 * 7)
        x = self.linear_block(x)
        return x


class AlternativeCNN1(nn.Module):
    def __init__(self, k):
        super().__init__()
        stride = 1 if k == 500 else 2 if k == 1000 else 3
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(4, 64, (1, ), (stride, )),  # shape: [n, 64, 500]
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2, 2),  # shape: [n, 64, 250]
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 256, (3, ), (4, )),  # shape: [n, 256, 62]
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2, 2),  # shape: [n, 256, 31]
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, 64, (5, ), (4, )),  # shape: [n, 64, 7]
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(7)  # shape: [n, 64, 1]
        )
        self.linear_block = nn.Sequential(
            nn.Linear(64, 2000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.reshape(-1, 64)
        x = self.linear_block(x)
        return x


class AlternativeCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=(9, ), stride=(3, ))  # shape: [n, 64, 164]
        self.b = nn.BatchNorm1d(64)
        self.g = nn.GELU()
        self.p = nn.MaxPool1d(2, 2)  # shape: [n, 64, 82]
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(3, ), stride=(2, )),  # shape: [n, 128, 40]
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2, 2)  # shape: [n, 128, 20]
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(5, ), stride=(2, )),  # shape: [n, 256, 8]
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2, 2)  # shape: [n, 256, 4]
        )
        self.linear_block = nn.Sequential(
            nn.Linear(256 * 4, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(.2),
            nn.Linear(64, 2000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.p(self.g(self.b(self.conv1(x))))
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.reshape(-1, 256 * 4)
        x = self.linear_block(x)
        return x

    def get_feature_map(self, x):
        return self.conv1(x)


def train(net, loader, cri, opt, device, max_epoch):
    net.train()
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


def test(net, loader, device):
    net.eval()
    res = []
    for idx, (data, label) in enumerate(loader):
        data = data.to(device)
        output = net(data)
        output = output.detach().cpu().numpy()
        res.append(output)
    res = np.concatenate(res, axis=0)
    return res


def get_conv_map(net, loader, device):
    net.eval()
    f_map_list = []
    for (data, label) in loader:
        net = net.eval()
        data = data.to(device)
        f_map = net.get_feature_map(data).detach().cpu().numpy()
        f_map_list.append(f_map)
    f_map_list = np.concatenate(f_map_list)
    return f_map_list


def get_motif(f_map, thr, data_str, data_int, k_size, stride):
    index = np.where(f_map > thr)
    batch_index, chan_index, seq_index = index
    print(len(batch_index))
    seq_index = seq_index * stride
    res_str = []
    res_int = []
    for batch in batch_index:
        for chan in chan_index:
            for seq in seq_index:
                res_str.append(data_str[batch][seq: seq+k_size])
                res_int.append(data_int[batch][seq: seq+k_size])
    return res_str, res_int
