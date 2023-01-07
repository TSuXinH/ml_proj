import random

import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import scanpy as sc 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import datetime
from torch.utils.tensorboard import SummaryWriter, writer

map_dict = {
    'A':[1,0,0,0],
    'G':[0,1,0,0],
    'C':[0,0,1,0],
    'T':[0,0,0,1]
}
file_path_template = 'D:/young_study/young_study/ML_computer/courseProject/data/{}/{}'# './{}/{}'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class Model(nn.Module):
    def __init__(self,input_size, hidden_dim, output_size, n_layers):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # RNN可以直接输入
        self.hidden, self.cell = self.init_hidden(batch_size), self.init_hidden(batch_size)
        self.lstm = nn.LSTM(
            input_size=input_size, # input_size是指的sequence每一个位置的维度
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            bidirectional=False)
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, out_features=output_size) ### 256 -> 2000 * 2
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x):
        b = x.shape[0]
        out, (hidden, cell) = self.lstm(x,(self.hidden[:, :b, :], self.cell[:, :b, :])) # ,hidden
        out = self.relu(out)
        out = self.fc(out)
        out = out[:, -1, :]
        out = out.reshape((out.shape[0], 2000, 2))
        return self.softmax(out).reshape([out.shape[0] * 2000, 2]), (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

class DNADataset(Dataset):
    def __init__(self, dna_length, train_or_test='train'): # , n_step=50
        with open((file_path_template+'/sequences_{}.txt').format(train_or_test, dna_length, train_or_test),'r') as f:
          x = f.readlines()
        self.x = np.array([[map_dict[tmp] for tmp in list(_.strip())] for _ in x])
        y = sc.read((file_path_template+'/matrix_{}.mtx').format(train_or_test, dna_length, train_or_test))
        self.y = y.X.todense()
        self.x = torch.from_numpy(self.x)
        self.x = self.x.to(device).float()# self.x.reshape((self.x.shape[0],1,self.x.shape[1])).to(device).float()
        # self.x = torch.transpose(self.x, 1, 2) # 交换两个维度
        self.y = torch.from_numpy(self.y).to(device).float()
        # self.n_step = n_step
        # self.x = self.x.reshape(shape=(self.x.shape[0], n_step, -1))
        """到这里，相当于把数据集准备成了[40000,500,4]和[40000,2000]的两个torch，分别是self.x和self.y
        
        """
        print("X Shape", self.x.shape)
        print("y Shape", self.y.shape)
        del x; del y
        assert self.x is not None
        assert self.y is not None

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TorchManager:
    def __init__(self):
        pass

    def train(self):
        train_step = 0
        for epoch in range(n_epochs):
            tqdm_epochs = tqdm(train_dataloader)
            for (x, y) in tqdm_epochs:
                x, y = x.reshape((x.shape[0], -1, input_size)), y.reshape([-1, 1])
                output,_ = model(x)
                onehot_y = torch.zeros((batch_size * 2000, class_num)).to(device).scatter_(1, y.long(), 1)

                loss = -torch.log(output * onehot_y.detach() + 1e-3).mean()

                # 梯度变化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 由于是二分类，因此将大于0.5的设置为1，小于0.5的设置为0
                # 为了计算accuracy，打平之后计算
                pred = torch.argmax(output, dim=1).float().reshape((-1, ))
                flatten_y = y.reshape((-1, ))
                accuracy = torch.count_nonzero(pred == flatten_y) / pred.shape[0]

                # tqdm显示Epoch和Acc
                tqdm_epochs.set_description('Epoch %i' % epoch)
                tqdm_epochs.set_postfix(loss=float(loss), accuracy=accuracy)

                ### tensorboard ###
                writer.add_scalar('loss/loss', loss, train_step)
                writer.add_scalar('evaluate/accuracy', accuracy, train_step)
                train_step += 1

            self.save_all()

    def evaluate(self):
        self.load()

        goal = model.fc.weight.reshape([2000, -1]).detach().cpu().numpy()
        print(goal.shape)
        np.save("filename.npy", goal)
        exit()


        tqdm_epochs = tqdm(test_dataloader)
        total_right0, total_count0, total_right1, total_count1 = 0, 0, 0, 0
        for train_step, (x, y) in enumerate(tqdm_epochs):
            x, y = x.reshape((x.shape[0], -1, input_size)), y.reshape([-1, 1])
            output, _ = model(x)

            # 由于是二分类，因此将大于0.5的设置为1，小于0.5的设置为0
            # 为了计算accuracy，打平之后计算

            pred = (output[:, 0] <= threshold * output[:, 1]).float().reshape((-1,))
            flatten_y = y.reshape((-1,))

            index_0 = flatten_y == 0
            index_1 = flatten_y == 1

            total_right0 += torch.count_nonzero(pred[index_0] == 0)
            total_count0 += torch.count_nonzero(index_0)

            total_right1 += torch.count_nonzero(pred[index_1] == 1)
            total_count1 += torch.count_nonzero(index_1)

            # print(total_right0, total_count0, total_right1, total_count1)

        print('Final 0 Accuracy {:.3f} 1 Accuracy {:.3f}'.format(total_right0 / total_count0, total_right1 / total_count1))

    def save_all(self):
        torch.save(model.state_dict(), "./model_v" + v)
        torch.save(optimizer.state_dict(), "./model_v" + v+ "_optimizer")

    def load(self):
        print('Begin Load Model...')
        model.load_state_dict(torch.load("./model_v" + v))
        # optimizer.load_state_dict(torch.load("./model_v" + v+ "_optimizer"))
        print('Load Model Over...')

if __name__ == '__main__':
    n_epochs = 100
    lr = 0.1
    dna_length = 500
    batch_size = 64
    class_num = 2
    step = 10
    input_size = step * 4
    threshold = 13

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # sequence_length=dna_length,
    model = Model(input_size=input_size, hidden_dim=256, output_size=2000 * 2, n_layers=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_dataset = DNADataset(dna_length=dna_length, train_or_test='train')
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    test_dataset = DNADataset(dna_length=dna_length, train_or_test='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    v = '1'
    # experiment_dir = './tensorboard/' + datetime.datetime.now().strftime("%m%d") + '_ml_v' + v + '/'
    # writer = SummaryWriter(experiment_dir + '-version' + v)

    torch_manager = TorchManager()
    # torch_manager.train()
    torch_manager.evaluate()
        