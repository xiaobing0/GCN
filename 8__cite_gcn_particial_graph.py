'''用引文数据库，拆两个子图，只针对一个子图训练，'''
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
import time
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

from dgl.data import citation_graph as citegrh

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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
    download_en(g)  # 下载, 针对 h，是将h加上服务器上的d
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


g, features, labels, train_mask, test_mask = load_cora_data()

# feature = feature.to(device)  # 放到gpu
# N = 10
# siz = feature[0].size().numel()//N  # 特征分成N分。每份大小
# features = torch.zeros(len(feature), siz).to(device)
# for i in range(len(feature)):
#     features[i][0:siz] = feature[i][0:siz]


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
list1 = random.sample(range(1, 2708), 1200)

# list1 = [6, 68, 137, 258, 339, 440, 555, 689, 767, 899, 988, 1233, 2344]
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

end = time.time()
print("前期运行时间:%.2f秒" % (end - start))

'''######################### 模型  #########################################'''

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


###############################################################################


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


###############################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


net = Net()
net = net.to(device)
# print(net)

###############################################################################
# We load the cora dataset using DGL's built-in data module.

from dgl.data import citation_graph as citegrh
import networkx as nx


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


###############################################################################


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


###############################################################################

optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
acc = []
for epoch in range(300):

    t0 = time.time()

    net.train()
    logits = net(g2, features2)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask2], labels2[train_mask2])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - t0)

    acc1 = evaluate(net, g2, features2, labels2, test_mask2)
    acc.append(acc1)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc1, np.mean(dur)))

import pandas as pd

acc = pd.DataFrame(data = acc)
acc.to_csv('./test.csv')