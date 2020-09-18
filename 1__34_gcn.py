import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import time
import numpy as np


def build_karate_club_graph():
    g = dgl.DGLGraph()
    # 添加34个节点
    g.add_nodes(34)
    # 通过元组列表添加78条边
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
    g.add_edges(src, dst)
    # 边是有方向的，并使他们双向
    g.add_edges(dst, src)

    return g


G = build_karate_club_graph()

'''Step 3: 定义一个图卷积神经网络（GCN）'''


# 定义 message function and reduce function
# NOTE: 在本教程中，我们将忽略GCN的规范化常数c_ij。
def gcn_message(edges):
    # 该函数批量处理边
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    # 该函数批量处理节点
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}


nx_G = G.to_networkx().to_undirected()



# 定义GCNLayer模块
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        h = g.ndata.pop('h')
        return self.linear(h)


# 一般来说，节点通过“message”函数传递信息，然后通过“reduce”函数进行数据聚合。
# 定义一个包含两个GCN layers的GCN模型
# 定义一个包含两个GCN layers的GCN模型

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


net = GCN(34, 5, 2)

'''Step 4: 数据准备和初始化'''
# 我们使用one-hot向量来初始化节点特征。 由于这是半监督设置，因此仅为教练（node 0）
# 和俱乐部主席（node 33）分配标签。 该实现如下:

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

'''Step 5: 训练然后可视化'''
# 训练循环和其他的 Pytorch 模型相同
# (1)创建一个优化函数, (2) 将输入提供给模型
# (3) 计算损失        (4) 使用autograd优化模型.

start = time.time()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(100):
    logits = net(G, inputs)
    # 我们保存the logits以便于接下来可视化
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # 我们仅仅为标记过的节点计算loss
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
end = time.time()

aa = logits.detach().numpy()
for i in range(34):
    b= aa[i].argmax()
    print('{} to {}'.format(i,b))
# ###############################################################################
# # This is a rather toy example, so it does not even have a validation or test
# # set. Instead, Since the model produces an output feature of size 2 for each node, we can
# # visualize by plotting the output feature in a 2D space.
# # The following code animates the training process from initial guess
# # (where the nodes are not classified correctly at all) to the end
# # (where the nodes are linearly separable).
#
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
print("循环运行时间:%.2f秒"%(end-start))
# ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
