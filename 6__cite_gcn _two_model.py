'''引文网络，针对一个图，每个模型分为两层子模型'''
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import torch
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###############################################################################
# We then define the node UDF for ``apply_nodes``, which is a fully-connected layer:

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

class Net1(nn.Module):  # 第一层
    def __init__(self):
        super(Net1, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)

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


net1 = Net1()  # net 的参数是g和inputs
net2 = Net2()  # net 的参数是g和inputs

net1 = net1.to(device)
net2 = net2.to(device)
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


def evaluate(model1, model2, g, features, labels, mask):
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


###############################################################################
# We then train the network as follows:

import time
import numpy as np

g, features, labels, train_mask, test_mask = load_cora_data()
# optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)


optimizer = torch.optim.Adam([
    {'params': net1.parameters(), 'lr': 1e-3},
    {'params': net2.parameters(), 'lr': 1e-3}
])
dur = []
for epoch in range(50):
    if epoch >= 3:
        t0 = time.time()

    net1.train()
    net2.train()
    logits1 = net1(g, features)
    logits = net2(g, logits1)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(net1,net2, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc, np.mean(dur)))

###############################################################################
