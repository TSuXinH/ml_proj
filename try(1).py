import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from munkres import Munkres,print_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import math

para = np.load("filename.npy")
print (para.shape)
f = open("D:/young_study/young_study/ML_computer/courseProject/data/celltype.txt", "r",encoding='utf-8')
label=[]

#print(type(line),line)
def cell_type(str):
    #change = []
    global change
    if str=='CLP\n':
        change=0
    if str=='CMP\n':
        change=1
    if str=='GMP\n':
        change=2
    if str=='HSC\n':
        change=3
    if str=='LMPP\n':
        change=4
    if str=='MEP\n':
        change=5
    if str=='MPP\n':
        change=6
    if str=='pDC\n':
        change=7
    if str=='UNK\n':
        change=8
    if str=='mono\n':
        change=9
    return change

line = f.readline() # 读取第一行
while line:
    txt_data = cell_type(line) # 可将字符串变为元组
    #print (txt_data)
    label.append(txt_data) # 列表增加
    line = f.readline() # 读取下一行

label=np.array(label)
print(label.shape)

#print(type_list)

#把聚类结果和实际标签映射
def best_map(L1,L2):
#L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

###kmeans聚类###
k_means = KMeans(n_clusters=10,random_state=10)
k_means.fit(para)
y_predict = k_means.predict(para)

print(y_predict.shape)
new_pred=best_map(label,y_predict)
print(new_pred.shape)

#计算正确率
bingo = 0
for i in range(2000):
    if new_pred[i]==label[i]:
        bingo = bingo+1
        # print('bingo=',bingo)
        # print('yes')

acc = bingo/2000
print('bingo=',bingo)
print('acc=',acc)
# print(y_predict)

###绘制kmeans散点图###
para_new = para[:,0:2]
print(para_new.shape)
fig, ax = plt.subplots()
types = []

# 如果以颜色表示缩写来定义颜色，这样类簇大时颜色可能重复较多
# c = ['b','c','y','r','g','m','w','k']
# colors = [c[i%len(c)] for i in range(clusters_number)]

# 随机生成使用16进制表示的颜色
colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(2)])
colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

for i, color in enumerate(colors):
    need_idx = np.where(y_predict == i)[0]
    ax.scatter(para_new[need_idx, 1], para_new[need_idx, 0], c=color, label=i)

plt.xlim(0.5, 0.7)
plt.ylim(0.4, 0.7)

legend = ax.legend(loc='upper right')
plt.savefig("cluster.png")

###计算NMI###
def NMI(A, B):
    # 样本点数
    A=np.array(A)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)  # 输出满足条件的元素的下标
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)  # Find the intersection of two arrays.
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat

if __name__ == '__main__':
    A = np.array(label)
    B = new_pred
    print('kmeans聚类结果：')
    print(A.shape,B.shape)
    print(NMI(A, B))
    print(metrics.normalized_mutual_info_score(A, B))  # 直接调用sklearn中的函数

###层次聚类###
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='average')
hc_predict = ac.fit(para).labels_
#print(hc_pred.shape)
hc_pred = best_map(label,hc_predict)

A = label
B = hc_pred
print('层次聚类结果：')
print(A.shape,B.shape)
print(NMI(A, B))
print(metrics.normalized_mutual_info_score(A, B))  # 直接调用sklearn中的函数

###GMM聚类###
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=10).fit(para)
gmm_predict = gmm.predict(para)
print(gmm_predict.shape)
gmm_pred = best_map(label,gmm_predict)
print(gmm_pred.shape)

A = label
B = gmm_pred
print('GMM聚类结果：')
print(A.shape,B.shape)
print(NMI(A, B))
print(metrics.normalized_mutual_info_score(A, B))  # 直接调用sklearn中的函数