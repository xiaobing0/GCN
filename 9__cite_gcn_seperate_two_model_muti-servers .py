'''用引文数据库，拆两个字图，通过中间交互，每个模型分为两层子模型, 模拟有多个服务器'''
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from tkinter import _flatten
import numpy as np
import time
import warnings
import dgl.function as fn
from dgl import DGLGraph
import torch
from dgl.data import citation_graph as citegrh
import random

warnings.filterwarnings("ignore", category=UserWarning)

from dgl.data import citation_graph as citegrh

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''#########################加载数据，构建子图#########################################'''

start = time.time()


def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask


def evaluate(model1, model2, g, features, labels, mask):
    g.update_all(gcn_msg, gcn_reduce)  # 聚合,针对 h
    dic11 = get_download_nodes(g)
    download_en(g, dic11)  # 下载, 针对 h，是将h加上服务器上的d
    model1.eval()
    model2.eval()
    with th.no_grad():
        logits = model1(g, features)
        logits = model2(g, logits)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


g, feature, labels, train_mask, test_mask = load_cora_data()
feature = feature.to(device)  # 放到gpu
N = 10
siz = feature[0].size().numel() // N  # 特征分成N分。每份大小
features = torch.zeros(len(feature), siz).to(device)
for i in range(len(feature)):
    features[i][0:siz] = feature[i][0:siz]  # 缩小维度

g.ndata['h'] = features
labels = labels.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

node_list = list(range(2708))  # 点集
src, dst = g.all_edges()
src = src.detach().numpy()
dst = dst.detach().numpy()
edge_list = list(zip(src, dst))  # 边集

# L = list(range(2708))  # 点集
# list1 = random.sample(range(1, 2708), 1354)

list1 = [0, 6, 68, 500]
list2 = list(set(node_list) - set(list1))

g1 = g.subgraph(list1)  # 构建子图架构
g1.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g2 = g.subgraph(list2)  # 构建子图架构
g2.copy_from_parent()  # 上面只是拓扑结构，现添加特征

train_mask1 = train_mask[list1]
train_mask2 = train_mask[list2]
labels1 = labels[list1]
labels2 = labels[list2]
features1 = features[list1]
features2 = features[list2]

test_mask1 = test_mask[list1]
test_mask2 = test_mask[list2]
# nx.draw(g1.to_networkx(), with_labels=True)  # 画图
# plt.show()

'''######################### 定义server #########################################'''


def get_download_nodes(g_pe):  # 根据划分的子图，以字典的形式返回每个子图需要连接的其他点，序号为原始图中点的序号,相当于每个子图要下载同步的点
    node_dic = {}
    list_pe = g_pe.parent_nid.numpy()
    list_pe = list(list_pe)  # 子图的节点在原图中的序列
    for i in list_pe:  # 一个点在本图中, 节点标签是子图在原图中的标签

        for j in list(set(node_list) ^ set(list_pe)):  # 另一个点不在本图中
            if (i, j) in edge_list:
                node_dic.setdefault(i, [])  # 为字典添加键
                node_dic.setdefault(i, []).append(j)  # 为字典添加值
            elif (j, i) in edge_list:
                node_dic.setdefault(i, [])  # 为字典添加键
                node_dic.setdefault(i, []).append(j)  # 为字典添加值
    return node_dic


dic1 = get_download_nodes(g1)  # g1需要连接的其他图上的节点，序号为原始图中点的序号
dic2 = get_download_nodes(g2)  # g2需要连接的其他图上的节点，序号为原始图中点的序号

list_common = list(dic1.values()) + list(dic2.values())  # 总的需要更新的点，即server上需要保存的点，序号为原始图中点的序号
list_common = _flatten(list_common)
list_common = list(set(list_common))  # 去除重复点
common_nodes = {}  # 将需要更新的点生成字典放到 server
for i in list_common:
    common_nodes.setdefault(i, [])


def get_upload_nodes(g_pe):  # 相当于每个子图需要上传到server更新的点
    list_up = []
    list_pe = g_pe.parent_nid.numpy()
    list_pe = list(list_pe)
    for i in list_pe:
        if i in list_common:  # 判断子图的节点是否被其他人引用
            list_up.append(i)
    return list_up


list_up1 = get_upload_nodes(g1)  # g1上传到server上的点
list_up2 = get_upload_nodes(g2)
#
'''########################定义密钥生成，加密，和解密函数################################'''


# def upload_en(g_pe, N):  # 每个子图加密上传到server,改变common_nodes字典里每个节点的值，第一层上传feat值保证字典里有值，从而可以下载到h
#     list_up = get_upload_nodes(g_pe)  # 得到该图中需要上传的点, 坐标是在原图中的坐标
#     siz = features[i].size().numel()//N  # 特征分成N分。每份大小
#     for j in N:
#         common_nodes[j] = {}  # 第一个键值为一个字典
#         for i in list_up:
#             if j != (N-1):
#                 a = features[i][j:j+siz]
#                 b = a
#                 common_nodes[j][i] = b  # 为对应的点 添加值
#             else:
#                 a = features[i][j:]
#                 b = a
#                 common_nodes[j][i] = b  # 为对应的点 添加值
#
#
# def upload_en1(g_pe, N):  # 第一轮以后，上传到字典里的是h的特征
#     list_up = get_upload_nodes(g_pe)  # 得到该图中需要上传的点
#     siz = features[i].size().numel() // N  # 特征分成N分。每份大小
#     for j in N:  # 分为N份
#         common_nodes[j] = {}  # 第一个键值为一个字典
#         for i in list_up:  # 每份都有所有的点，每份拿一个点的不同维度的特征
#             if j != (N - 1):
#                 a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'][j:j + siz]
#                 b = a
#                 common_nodes[j][i] = b  # 为对应的点 添加值
#             else:
#                 a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'][j:]
#                 b = a
#                 common_nodes[j][i] = b  # 为对应的点 添加值
#
#
# upload_en(g1)  # 加密上传
# upload_en(g2)
#
#
# def download_en(g_pe, N):  # 从server的字典里面拿到加密的数据,解密并更新子图
#     node_dic = get_download_nodes(g_pe)  # 得到该图需要下载同步的字典信息
#     a = list(node_dic.keys())  # 该子图中需要下载更新的节点
#     for i in a:
#         b = list(node_dic[i])  # 子图中和i节点相连的其他节点
#         s = g_pe.ndata['h'][0].size()  # 该图特征维度，维度随着层数而变换
#         sum_en = torch.zeros(s).to(device)
#         for j in b:
#             sum_link = [].to(device)  # 拼接的和
#             for m in N:  # 将N个部分拼接起来
#                 sum_link = np.vstack((sum_link, common_nodes[m][j].to(device)))
#             sum_en = sum_en + sum_link  # 把server中与i点相连的加密特征 相加
#         d = sum_en
#         g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] + d
#     return g_pe

def generate_key(w, m, n):
    S = (np.random.rand(m, n) * w / (2 ** 16))  # 可证明 max(S) < w
    return S  # key，对称加密


def encrypt(x, S, m, n, w):
    assert len(x) == len(S)
    e = (np.random.rand(m))  # 可证明 max(e) < w / 2
    c = np.linalg.inv(S).dot((w * x) + e)
    return c


def decrypt(c, S, w):
    return (S.dot(c) / w).astype('int')


m = 5
n = m
w = 16
S = generate_key(w, m, n)


def upload_en(g_pe, list_up):  # 每个子图加密上传到server,改变common_nodes字典里每个节点的值，第一层上传feat值保证字典里有值，从而可以下载到h
    start_up = time.time()
    for i in list_up:
        # a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['feat']  # 对应到子图上的节点
        a = features[i]
        # b = encrypt(a, S, m, n, w)  # 加密
        b = a
        common_nodes[i] = b  # 为对应的点 添加值
    end_up = time.time()
    print("上传时间：%s" % (end_up - start_up))


def upload_en1(g_pe, list_up):  # 第一轮以后，上传到字典里的是h的特征
    start_up = time.time()
    for i in list_up:
        a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h']  # 对应到子图上的节点
        # print('a:%s' % a)
        a = a[0]
        # print('a[0]:%s' % a)
        # b = encrypt(a, S, m, n, w)  # 加密
        b = a
        common_nodes[i] = b  # 为对应的点 添加值
    end_up = time.time()
    print("上传时间：%s" % (end_up - start_up))


upload_en(g1, list_up1)  # 加密上传
upload_en(g2, list_up2)


def download_en(g_pe, node_dic):  # 从server的字典里面拿到加密的数据,解密并更新子图
    start_dow = time.time()
    a = list(node_dic.keys())  # 该子图中需要下载更新的节点
    for i in a:
        b = list(node_dic[i])  # 子图中和i节点相连的其他节点
        s = g_pe.ndata['h'][0].size()  # 该图特征维度，维度随着层数而变换
        # print('下载维度：%s'%s)
        sum_en = torch.zeros(s).to(device)
        for j in b:
            sum_en = sum_en + common_nodes[j].to(device)  # 把server中与i点相连的加密特征 相加
        # d = decrypt(sum_en, S, w)  # 解密
        d = sum_en
        # d = torch.tensor(d)
        g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] + d
    end_dow = time.time()
    print("下载时间：%s" % (end_dow - start_dow))
    return g_pe


'''####################################定义网络结构#####################################'''

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)

        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        # print(self.linear.weight)  # 线性变换的参数
        # print(self.linear.bias)
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}  # 将中间值h赋值给h


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature

        # g.update_all(gcn_msg, gcn_reduce)  # 聚合,针对 h
        # download_en(g)  # 下载, 针对 h，是将h加上服务器上的d
        g.apply_nodes(func=self.apply_mod)  # 线性变换, 针对 h
        # print(self.apply_mod.linear.weight)  # 线性变换的参数
        # print("nadta2:%s" % g.ndata)
        return g.ndata['h']


class Net1(nn.Module):  # 第一层
    def __init__(self):
        super(Net1, self).__init__()
        self.gcn1 = GCN(143, 16, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)  # 第一层
        return x


class Net2(nn.Module):  # 第二层
    def __init__(self):
        super(Net2, self).__init__()
        self.gcn2 = GCN(16, 7, None)

    def forward(self, g, x):
        x = self.gcn2(g, x)  # 第二层
        return x


net_A1 = Net1()  # A方第一层
net_A2 = Net2()  # A方第二层

net_B1 = Net1()  # B方第一层
net_B2 = Net2()  # B方第二层

net_A1 = net_A1.to(device)
net_A2 = net_A2.to(device)
net_B1 = net_B1.to(device)
net_B2 = net_B2.to(device)

'''##############################Step 5: 训练然后可视化###################'''
# (1)创建一个优化函数   (2) 将输入提供给模型   (3) 计算损失   (4) 使用auto-grad优化模型.

optimizer1 = torch.optim.Adam([
    {'params': net_A1.parameters(), 'lr': 1e-2},
    {'params': net_A2.parameters(), 'lr': 1e-2}
])

optimizer2 = torch.optim.Adam([
    {'params': net_B1.parameters(), 'lr': 1e-2},
    {'params': net_B2.parameters(), 'lr': 1e-2}
])

dur = []
all_logits1 = []
all_logits2 = []

end = time.time()
print("前期运行时间:%.2f秒" % (end - start))

for epoch in range(50):
    print(epoch)

    t0 = time.time()

    net_A1.train()
    net_B1.train()
    net_A2.train()
    net_B2.train()

    g1.update_all(gcn_msg, gcn_reduce)  # A聚合,针对 h
    download_en(g1, dic1)  # 下载, 针对 h，是将h加上服务器上的d

    logits_A1 = net_A1(g1, g1.ndata['h'])  # A的第一层

    g2.update_all(gcn_msg, gcn_reduce)  # B聚合,针对 h
    download_en(g2, dic2)  # 下载, 针对 h，是将h加上服务器上的d

    logits_B1 = net_B1(g2, g2.ndata['h'])  # B的第一层

    upload_en1(g1, list_up1)  # 第一个图同步上传，传到特征h
    upload_en1(g2, list_up2)  # 第二个图同步上传
    # print('第一层后%s' % common_nodes[0].size())

    #############  第二层  ##################

    g1.update_all(gcn_msg, gcn_reduce)  # A聚合,针对 h
    download_en(g1, dic1)  # 下载, 针对 h，是将h加上服务器上的d
    # print(g1.ndata['h'].size())
    logits1 = net_A2(g1, logits_A1)  # A的第二层

    g2.update_all(gcn_msg, gcn_reduce)  # B聚合,针对 h
    download_en(g2, dic2)  # 下载, 针对 h，是将h加上服务器上的d

    logits2 = net_B2(g2, logits_B1)  # B的第二层

    upload_en(g1, list_up1)  # 重新给服务器上的字典赋值初始特征
    upload_en(g2, list_up2)
    # print('还原%s' % common_nodes[0].size())

    g1.ndata['h'] = features1
    g2.ndata['h'] = features2

    all_logits1.append(logits1.detach())  # 记录
    logp1 = F.log_softmax(logits1, 1)
    loss1 = F.nll_loss(logp1[train_mask1], labels1[train_mask1])

    all_logits2.append(logits2.detach())  # 记录
    logp2 = F.log_softmax(logits2, 1)
    loss2 = F.nll_loss(logp2[train_mask2], labels2[train_mask2])

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss1.backward(retain_graph=True)
    loss2.backward()
    optimizer1.step()
    optimizer2.step()

    p1 = net_A1.state_dict()  # 拿到net1参数字典
    p2 = net_B1.state_dict()  # 拿到net2参数字典
    for key, value in p2.items():  # p1等于两个字典平均
        p1[key] = (p1[key] + value) / 2

    net_A1.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B1.load_state_dict(p1)  # net2的参数更新为平均参数

    p1 = net_A2.state_dict()  # 拿到net1参数字典
    p2 = net_B2.state_dict()  # 拿到net2参数字典
    for key, value in p2.items():  # p1等于两个字典平均
        p1[key] = (p1[key] + value) / 2

    net_A2.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B2.load_state_dict(p1)  # net2的参数更新为平均参数

    dur.append(time.time() - t0)
    loss = (loss1 + loss2) / 2

    acc1 = evaluate(net_A1, net_A2, g1, features1, labels1, test_mask1)
    acc2 = evaluate(net_B1, net_B2, g2, features2, labels2, test_mask2)
    acc = (acc1 + acc2) / 2
    g1.ndata['h'] = features1
    g2.ndata['h'] = features2

    # print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
    #     epoch, loss.item(), np.mean(dur)))

    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc, np.mean(dur)))
