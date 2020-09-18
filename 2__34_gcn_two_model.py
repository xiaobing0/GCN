'''用34个点的图网络，针对一个图，把模型拆分成两个部分,即第一层和第二层分开'''

import dgl
import torch
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

warnings.filterwarnings("ignore", category=UserWarning)

'''###########################构建和划分子图###########################'''
G = dgl.DGLGraph()
# 添加34个节点
G.add_nodes(34)
# 通过元组列表添加78条边
node_list = list(range(34))

edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
             (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
             (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
             (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
             (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
             (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
             (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
             (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
             (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
             (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
             (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
             (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
             (33, 31), (33, 32)]
# 为边添加两个列表：src and dst
src, dst = tuple(zip(*edge_list))
G.add_edges(src, dst)
# 边是有方向的，并使他们双向
G.add_edges(dst, src)

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

'''###########################定义模型###########################'''
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
nx_G = G.to_networkx().to_undirected()


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


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)

        return g.ndata.pop('h')


class Net1(nn.Module):  # 第一层
    def __init__(self):
        super(Net1, self).__init__()
        self.gcn1 = GCN(34, 5, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)  # 第一层
        # print(self.gcn1.apply_mod.linear.weight)  # 第一层线性变换的参数，也是会根据训练变换的
        return x


class Net2(nn.Module):  # 第二层
    def __init__(self):
        super(Net2, self).__init__()
        self.gcn2 = GCN(5, 2, None)

    def forward(self, g, x):
        x = self.gcn2(g, x)  # 第二层
        return x


net1 = Net1()  # net 的参数是g和inputs
net2 = Net2()  # net 的参数是g和inputs

'''########################### 训练 ###########################'''
# (1)创建一个优化函数   (2) 将输入提供给模型   (3) 计算损失   (4) 使用auto-grad优化模型.


optimizer = torch.optim.Adam([
    {'params': net1.parameters(), 'lr': 0.01},
    {'params': net2.parameters(), 'lr': 0.01}
])

all_logits = []
for epoch in range(30):
    logits1 = net1(G, inputs)
    logits = net2(G, logits1)
    all_logits.append(logits.detach())  # 我们保存the logits以便于接下来可视化
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[labeled_nodes], labels)  # 我们仅仅为标记过的节点计算loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

'''##############################画图显示分类趋势##########################################'''
aa = logits.detach().numpy()
for i in range(34):
    b = aa[i].argmax()
    print('{} to {}'.format(i, b))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
                     with_labels=True, node_size=300, ax=ax)


fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(24)  # draw the prediction of the first epoch
# plt.close()

# ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
