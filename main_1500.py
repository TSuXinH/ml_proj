import os
import numpy as np
import scanpy as sc
from copy import deepcopy
import umap
import warnings
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from util import transfer_letter_to_num, cal_auc
from loader import CustomDataset
from net import CNN, train, test

warnings.filterwarnings('ignore')


path_train = '/data/train'
path_test = './data/test'
path_cell_type = './data/celltype.txt'
train_path_list = []
test_path_list = []
for idx in ['train', 'test']:
    for item in [500, 1500, 1500]:
        if idx == 'train':
            train_path_list.append('data/train/{}/sequences_train.txt'.format(item))
        else:
            test_path_list.append('data/test/{}/sequences_test.txt'.format(item))

with open(train_path_list[2], 'r') as f:
    train_1500 = f.readlines()

with open(test_path_list[2], 'r') as f:
    test_1500 = f.readlines()

with open(path_cell_type, 'r') as f:
    cell = f.readlines()


f_mat_train_1500 = sc.read('./data/train/1500/matrix_train.mtx')
label_train_1500 = np.array(f_mat_train_1500.X.todense()).astype(np.int_)

f_mat_test_1500 = sc.read('./data/test/1500/matrix_test.mtx')
label_test_1500 = np.array(f_mat_test_1500.X.todense()).astype(np.int_)

mat_train_1500 = transfer_letter_to_num(train_1500)
mat_test_1500 = transfer_letter_to_num(test_1500)


max_epoch = 30
batch_size = 256
lr = 1e-4
wd = 5e-4
device = 'cuda'


train_dataset = CustomDataset(mat_train_1500, label_train_1500)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CustomDataset(mat_test_1500, label_test_1500)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
net = CNN(k=1500).to(device)
cri = nn.BCELoss().to(device)
opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

loss_list = train(net, train_loader, cri, opt, device, max_epoch)


plt.plot(loss_list)
plt.title('training loss via epoch')
plt.show(block=True)


pred = test(net, test_loader, label_test_1500, device)
auc_list, auc = cal_auc(pred, label_test_1500)

plt.hist(auc_list, bins=50)
plt.title('auc distribution')
plt.show(block=True)

clus_data = net.linear_block[3].weight.cpu().detach().numpy()
clus_data = clus_data.reshape(2000, -1)


cell_str = ''.join(cell)
cell_str = cell_str.replace('\n', '')
cell_str = cell_str.replace('CLP', '0')
cell_str = cell_str.replace('CMP', '1')
cell_str = cell_str.replace('GMP', '2')
cell_str = cell_str.replace('HSC', '3')
cell_str = cell_str.replace('LMPP', '4')
cell_str = cell_str.replace('MEP', '5')
cell_str = cell_str.replace('MPP', '6')
cell_str = cell_str.replace('pDC', '7')
cell_str = cell_str.replace('UNK', '8')
cell_str = cell_str.replace('mono', '9')
cell_int = np.array([int(s) for s in cell_str])


clus_k = KMeans(n_clusters=10)
clus_res = clus_k.fit_predict(clus_data)


tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto')
dim_rdc_res = tsne.fit_transform(clus_data)
# u = umap.UMAP()
# dim_rdc_res = u.fit_transform(clus_data)


for item in range(np.max(clus_res) + 1):
    index = np.where(clus_res == item)[0]
    tmp_cell = dim_rdc_res[index]
    plt.scatter(tmp_cell[:, 0], tmp_cell[:, 1], label=item, s=8)
plt.legend()
plt.show(block=True)
