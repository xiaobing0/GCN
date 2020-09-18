'''用一个小型图网络,5个点， 图拆两个字图，通过中间交互，每部分图模型拆两部分，第一层和第二层，每层同步'''
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import _flatten
import numpy as np
import matplotlib.pyplot as plt
import dgl.function as fn
import networkx as nx
import torch.nn as nn

from dgl import DGLGraph
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


g = dgl.DGLGraph()
g.add_nodes(5)  # 在图中添加5个节点，分别标记为0至4
node_list = list(range(5))

edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
             (4, 0)]

g.ndata['feat'] = torch.eye(5).to(device)  # 特征初始化
g.ndata['h'] = g.ndata['feat']

src, dst = tuple(zip(*edge_list))
g.add_edges(src, dst)
g.add_edges(dst, src)

list1 = [0, 1, 3]  # 每个子图构成的节点
list2 = [2, 4]
g1 = g.subgraph(list1)  # 生成子图
g1.copy_from_parent()  # 为子图添加节点特征

g2 = g.subgraph(list2)
g2.copy_from_parent()

labeled_nodes1 = torch.tensor([0]).to(device)  # 选择标签节点
labels1 = torch.tensor([0]).to(device)  # 分配标签

labeled_nodes2 = torch.tensor([4]).to(device)  # 选择标签节点
labels2 = torch.tensor([1]).to(device)  # 分配标签

labeled_nodes2 = g2.map_to_subgraph_nid(labeled_nodes2)

'''###############################################################'''


def get_download_nodes(g_pe):  # 根据划分的子图，以字典的形式返回每个子图需要连接的其他点，序号为原始图中点的序号,相当于每个子图要下载同步的点
    node_dic = {}
    list_pe = g_pe.parent_nid.numpy()
    list_pe = list(list_pe)  # 子图的节点在原图中的序列
    for i in list_pe:  # 一个点在本图中, 节点标签是子图在原图中的标签
        node_dic.setdefault(i, [])  # 为字典添加键
        for j in list(set(node_list) ^ set(list_pe)):  # 另一个点不在本图中
            if (i, j) in edge_list:
                node_dic.setdefault(i, []).append(j)  # 为字典添加值
            if (j, i) in edge_list:
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

'''########################定义密钥生成，加密，和解密函数################################'''


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


def upload_en(g_pe):  # 每个子图加密上传到server,改变common_nodes字典里每个节点的值，第一层上传feat值保证字典里有值，从而可以下载到h
    list_up = get_upload_nodes(g_pe)  # 得到该图中需要上传的点
    for i in list_up:
        a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['feat']  # 对应到子图上的节点
        # print('a:%s'% a)
        a = a[0]
        # b = encrypt(a, S, m, n, w)  # 加密
        # print('a[0]:%s'% a)
        b = a
        common_nodes[i] = b  # 为对应的点 添加值


def upload_en1(g_pe):  # 第一轮以后，上传到字典里的是h的特征
    list_up = get_upload_nodes(g_pe)  # 得到该图中需要上传的点
    for i in list_up:
        a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h']  # 对应到子图上的节点
        a = a[0]
        # b = encrypt(a, S, m, n, w)  # 加密
        b = a
        common_nodes[i] = b  # 为对应的点 添加值


upload_en(g1)  # 加密上传
upload_en(g2)


def download_en(g_pe):  # 从server的字典里面拿到加密的数据,解密并更新子图
    node_dic = get_download_nodes(g_pe)  # 得到该图需要下载同步的字典信息
    a = list(node_dic.keys())  # 该子图中需要下载更新的节点
    for i in a:
        b = list(node_dic[i])  # 子图中和i节点相连的其他节点
        s = g_pe.ndata['h'][0].size()  # 该图特征维度，维度随着层数而变换
        sum_en = torch.zeros(s).to(device)
        for j in b:
            sum_en = sum_en + common_nodes[j].to(device)  # 把server中与i点相连的加密特征 相加
        # d = decrypt(sum_en, S, w)  # 解密
        d = sum_en
        # d = torch.tensor(d)
        g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] + d
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
        return {'h': h}  # 将中间值h赋值给feat


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
        self.gcn1 = GCN(5, 3, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)  # 第一层
        return x


class Net2(nn.Module):  # 第二层
    def __init__(self):
        super(Net2, self).__init__()
        self.gcn2 = GCN(3, 2, None)

    def forward(self, g, x):
        x = self.gcn2(g, x)  # 第二层
        return x


net_A1 = Net1() # A方第一层
net_A2 = Net2()  # A方第二层

net_B1 = Net1()  # B方第一层
net_B2 = Net2()  # B方第二层

net_A1 = net_A1.to(device)
net_A2 = net_A2.to(device)
net_B1 = net_B1.to(device)
net_B2 = net_B2.to(device)

'''Step 5: 训练然后可视化'''
# (1)创建一个优化函数   (2) 将输入提供给模型   (3) 计算损失   (4) 使用auto-grad优化模型.

optimizer1 = torch.optim.Adam([
    {'params': net_A1.parameters(), 'lr': 1e-2},
    {'params': net_A2.parameters(), 'lr': 1e-2}
])

optimizer2 = torch.optim.Adam([
    {'params': net_B1.parameters(), 'lr': 1e-2},
    {'params': net_B2.parameters(), 'lr': 1e-2}
])

all_logits1 = []
all_logits2 = []
for epoch in range(1):
    print(epoch)

    g1.update_all(gcn_msg, gcn_reduce)  # 聚合,针对 h
    download_en(g1)  # 下载, 针对 h，是将h加上服务器上的d

    logits_A1 = net_A1(g1, g1.ndata['h'])  # A的第一层

    g2.update_all(gcn_msg, gcn_reduce)  # 聚合,针对 h
    download_en(g2)  # 下载, 针对 h，是将h加上服务器上的d

    logits_B1 = net_B1(g2, g2.ndata['h'])  # B的第一层
    upload_en1(g1)  # 第一个图同步上传，传到特征h
    upload_en1(g2)  # 第二个图同步上传

    g1.update_all(gcn_msg, gcn_reduce)  # 聚合,针对 h
    download_en(g1)  # 下载, 针对 h，是将h加上服务器上的d
    print(g1.ndata['h'])
    logits1 = net_A2(g1, logits_A1)  # A的第二层

    g2.update_all(gcn_msg, gcn_reduce)  # 聚合,针对 h
    download_en(g2)  # 下载, 针对 h，是将h加上服务器上的d

    logits2 = net_B2(g2, logits_B1)  # B的第二层

    upload_en(g1)  # 重新给服务器上的字典赋值初始特征
    upload_en(g2)

    g1.ndata['h'] = g1.ndata['feat']
    g2.ndata['h'] = g2.ndata['feat']

    all_logits1.append(logits1.detach())  # 记录
    logp1 = F.log_softmax(logits1, 1)
    loss1 = F.nll_loss(logp1[labeled_nodes1], labels1)

    all_logits2.append(logits2.detach())  # 记录
    logp2 = F.log_softmax(logits2, 1)
    loss2 = F.nll_loss(logp2[labeled_nodes2], labels2)

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss1.backward(retain_graph = True)
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

    # p1 = list(net.named_parameters())[0][1]
    # print('Epoch %d | Loss: %.4f' % (epoch, loss2.item()))
    print('Epoch %d | Loss: %.4f' % (epoch, (loss1.item() + loss2.item()) / 2))

'''##############################画图显示分类趋势##########################################'''

aa = logits1.detach().cpu().numpy()
for i in range(3):
    b = aa[i].argmax()
    print('{} to {}'.format(g1.parent_nid.numpy()[i], b))
nx_G = g1.to_networkx().to_undirected()


def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(3):
        pos[v] = all_logits1[i][v].cpu().numpy()
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
draw(29)
# ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)

