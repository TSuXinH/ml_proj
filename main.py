import os
import numpy as np
import scanpy as sc
from copy import deepcopy
import umap.umap_ as umap
import warnings
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from util import transfer_letter_to_num, cal_auc
from loader import CustomDataset
from net import CNN, train, test, AlternativeCNN, AlternativeCNN1

warnings.filterwarnings('ignore')


path_train = '/data/train'
path_test = './data/test'
path_cell_type = './data/celltype.txt'
train_path_list = []
test_path_list = []
for idx in ['train', 'test']:
    for item in [500, 1000, 1500]:
        if idx == 'train':
            train_path_list.append('data/train/{}/sequences_train.txt'.format(item))
        else:
            test_path_list.append('data/test/{}/sequences_test.txt'.format(item))

with open(train_path_list[0], 'r') as f:
    train_500 = f.readlines()

with open(test_path_list[0], 'r') as f:
    test_500 = f.readlines()

with open(path_cell_type, 'r') as f:
    cell = f.readlines()


f_mat_train_500 = sc.read('./data/train/500/matrix_train.mtx')
label_train_500 = np.array(f_mat_train_500.X.todense()).astype(np.int_)

f_mat_test_500 = sc.read('./data/test/500/matrix_test.mtx')
label_test_500 = np.array(f_mat_test_500.X.todense()).astype(np.int_)

mat_train_500 = transfer_letter_to_num(train_500)
mat_test_500 = transfer_letter_to_num(test_500)


max_epoch = 30
batch_size = 256
lr = 1e-4
wd = 5e-4
device = 'cuda'


train_dataset = CustomDataset(mat_train_500, label_train_500)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CustomDataset(mat_test_500, label_test_500)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
net = AlternativeCNN(k=500).to(device)
cri = nn.BCELoss().to(device)
opt = optim.RAdam(net.parameters(), lr=lr, weight_decay=wd)

loss_list = train(net, train_loader, cri, opt, device, max_epoch)


plt.plot(loss_list)
plt.title('training loss via epoch')
plt.show(block=True)


pred = test(net, test_loader, label_test_500, device)
auc_list, auc = cal_auc(pred, label_test_500)

plt.hist(auc_list, bins=50)
plt.title('auc distribution')
plt.show(block=True)

clus_data = net.linear_block[4].weight.cpu().detach().numpy()
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

res_list = []
for item in range(np.max(cell_int)+1):
    index = np.where(cell_int == item)[0]
    print(item, np.argmax(np.bincount(clus_res[index])))
    res_list.append(clus_res[index])

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
dim_rdc_res = tsne.fit_transform(clus_data)


for item in range(np.max(clus_res) + 1):
    index = np.where(cell_int == item)[0]
    tmp_cell = dim_rdc_res[index]
    plt.scatter(tmp_cell[:, 0], tmp_cell[:, 1], label=item, s=10)
plt.legend()
plt.show(block=True)

ami = adjusted_mutual_info_score(cell_int, clus_res)
nmi = normalized_mutual_info_score(cell_int, clus_res)
