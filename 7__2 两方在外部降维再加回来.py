'''用引文数据库，拆两个字图，通过中间交互，每个模型分为两层子模型'''
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
from dgl.data import citation_graph as citegrh
from funcs import load_cora_data, Net1, Net2, Net3, evaluate


warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cpu')
'''#########################加载数据，构建子图#########################################'''
start = time.time()

g, features, labels, train_mask, test_mask = load_cora_data()
g.ndata['h'] = features
node_list = list(range(2708))  # 点集
src, dst = g.all_edges()
src = src.detach().numpy()
dst = dst.detach().numpy()
edge_list = list(zip(src, dst))  # 边集


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


def get_upload_nodes(g_pe):  # 相当于每个子图需要上传到server更新的点
    list_up = []
    list_pe = g_pe.parent_nid.numpy()
    list_pe = list(list_pe)
    for i in list_pe:
        if i in list_common:  # 判断子图的节点是否被其他人引用
            list_up.append(i)
    return list_up


def upload_en(g_pe, list_up):  # 每个子图加密上传到server,改变common_nodes字典里每个节点的值，第一层上传feat值保证字典里有值，从而可以下载到h
    for i in list_up:
        # a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['feat']  # 对应到子图上的节点
        a = features[i]
        # b = crypten.cryptensor(a)  # 加密上传
        common_nodes[i] =a  # 为对应的点 添加值


def upload_en1(g_pe, list_up):  # 第一轮以后，上传到字典里的是h的特征
    for i in list_up:
        a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h']  # 对应到子图上的节点
        a = a[0]
        # b = crypten.cryptensor(a)
        common_nodes[i] = a  # 为对应的点 添加值


def download_en(g_pe, node_dic, par):  # 从server的字典里面拿到加密的数据,解密并更新子图
    a = list(node_dic.keys())  # 该子图中需要下载更新的节点
    aa, bb = par.size()  # 16,1433
    for i in a:
        b = list(node_dic[i])  # 子图中和i节点相连的其他节点
        s = g_pe.ndata['h'][0].size()  # 该图特征维度，维度随着层数而变换
        sum_en = torch.zeros(s).to(device)
        for j in b:  # 对每一个邻居
            sum_en_sub = torch.zeros(aa).to(device)  # 16个数，每个点16个特征
            # sum_en_sub = crypten.cryptensor(sum_en_sub)
            for m in range(aa):
                sum_en_sub[m] = sum(par[m] * common_nodes[j])
            # sum_en_sub = sum_en_sub.get_plain_text()  # 解密

            sum_en = sum_en + sum_en_sub.to(device)  # 把server中与i点相连的加密特征 相加
        d = sum_en
        g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] + d
    return g_pe


# L = list(range(2708))  # 点集
# list1 = random.sample(range(1, 2708), 1000)

list1 = [0, 6, 500]
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

'''######################### 定义server #########################################'''
dic1 = get_download_nodes(g1)  # g1需要连接的其他图上的节点，序号为原始图中点的序号
dic2 = get_download_nodes(g2)  # g2需要连接的其他图上的节点，序号为原始图中点的序号
#
list_common = list(dic1.values()) + list(dic2.values())  # 总的需要更新的点，即server上需要保存的点，序号为原始图中点的序号
list_common = _flatten(list_common)
list_common = list(set(list_common))  # 去除重复点
common_nodes = {}  # 将需要更新的点生成字典放到 server
for i in list_common:
    common_nodes.setdefault(i, [])

list_up1 = get_upload_nodes(g1)  # g1上传到server上的点
list_up2 = get_upload_nodes(g2)
#
'''########################定义密钥生成，加密，和解密函数################################'''
upload_en(g1, list_up1)  # 加密上传
upload_en(g2, list_up2)
'''####################################定义网络结构#####################################'''
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

net_A1 = Net1()  # A方第一层的线性变换，需要拿到参数上传
net_A2 = Net2()  # A方第一层后的激活层
net_A3 = Net3()  # A方第二层

net_B1 = Net1()  # B方第一层
net_B2 = Net2()  # B方第二层
net_B3 = Net3()

'''##############################Step 5: 训练然后可视化###################'''
# (1)创建一个优化函数   (2) 将输入提供给模型   (3) 计算损失   (4) 使用auto-grad优化模型.

optimizer1 = torch.optim.Adam([
    {'params': net_A1.parameters(), 'lr': 1e-1},
    {'params': net_A2.parameters(), 'lr': 1e-1},
    {'params': net_A3.parameters(), 'lr': 1e-1}
])

optimizer2 = torch.optim.Adam([
    {'params': net_B1.parameters(), 'lr': 1e-1},
    {'params': net_B2.parameters(), 'lr': 1e-1},
    {'params': net_B3.parameters(), 'lr': 1e-1}
])

dur = []
all_logits1 = []
all_logits2 = []
acc = []
end = time.time()
print("前期运行时间:%.2f秒" % (end - start))

for epoch in range(100):
    print(epoch)

    t0 = time.time()

    net_A1.train()
    net_B1.train()
    net_A2.train()
    net_B2.train()
    net_A3.train()
    net_B3.train()

    g1.update_all(gcn_msg, gcn_reduce)  # A聚合,针对 h
    logits_A1 = net_A1(g1, g1.ndata['h'])  # A的第一层,线性变换
    for parameters in net_A1.parameters():  # 线性变换的参数
        par1 = parameters
    # par1 = crypten.cryptensor(par1)  # 参数加密上传
    download_en(g1, dic1, par1)  # 下载, 针对 h，是将h加上服务器上的加密值乘以参数再解密
    logits_A1 = net_A2(g1, logits_A1)  # Relu

    g2.update_all(gcn_msg, gcn_reduce)  # B聚合,针对 h
    logits_B1 = net_B1(g2, g2.ndata['h'])  # B的第一层
    for parameters in net_B1.parameters():  # 线性变换的参数
        par2 = parameters
    # par2 = crypten.cryptensor(par2)  # 参数加密上传
    download_en(g2, dic2, par2)  # 下载, 针对 h，是将h加上服务器上的d
    logits_B1 = net_B2(g2, logits_B1)  # relu

    ###
    upload_en1(g1, list_up1)  # 第一个图同步上传，传到特征h
    upload_en1(g2, list_up2)  # 第二个图同步上传
    # print('第一层后%s' % common_nodes[0].size())

    #############44444444444444  第二层 444444444 ##################

    g1.update_all(gcn_msg, gcn_reduce)  # A聚合,针对 h
    logits1 = net_A3(g1, logits_A1)  # A的第二层
    for parameters in net_A3.parameters():  # 线性变换的参数
        par3 = parameters
    # par3 = crypten.cryptensor(par3)  # 参数加密上传
    download_en(g1, dic1, par3)  # 下载, 针对 h，是将h加上服务器上的d
    # print(g1.ndata['h'].size())

    g2.update_all(gcn_msg, gcn_reduce)  # B聚合,针对 h
    logits2 = net_B3(g2, logits_B1)  # B的第二层
    for parameters in net_B3.parameters():  # 线性变换的参数
        par4 = parameters
    # par4 = crypten.cryptensor(par4)  # 参数加密上传
    download_en(g2, dic2, par4)  # 下载, 针对 h，是将h加上服务器上的d

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

    dur.append(time.time() - t0)

    p1 = net_A1.state_dict()  # 拿到net1参数字典
    p2 = net_B1.state_dict()  # 拿到net2参数字典
    for key, value in p2.items():  # p1等于两个字典平均
        p1[key] = (p1[key] + value)/2

    net_A1.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B1.load_state_dict(p1)  # net2的参数更新为平均参数

    # p1 = net_A2.state_dict()  # 拿到net1参数字典
    # p2 = net_B2.state_dict()  # 拿到net2参数字典
    # for key, value in p2.items():  # p1等于两个字典平均
    #     p1[key] = (p1[key]*(len(list1)/2708) + value*(len(list2)/2708)) / 2

    # net_A2.load_state_dict(p1)  # net1的参数更新为平均参数
    # net_B2.load_state_dict(p1)  # net2的参数更新为平均参数

    p1 = net_A3.state_dict()  # 拿到net1参数字典
    p2 = net_B3.state_dict()  # 拿到net2参数字典
    for key, value in p2.items():  # p1等于两个字典平均
        p1[key] = (p1[key] + value)/2

    net_A3.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B3.load_state_dict(p1)  # net2的参数更新为平均参数

    loss = (loss1 + loss2) / 2

    g1.ndata['h'] = features1
    g2.ndata['h'] = features2

    upload_en(g1, list_up1)  # 重新给服务器上的字典赋值初始特征
    upload_en(g2, list_up2)

    acc1 = evaluate(net_A1, net_A2, net_A3, g, features, labels, test_mask)
    acc.append(acc1)
    # print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
    #     epoch, loss.item(), np.mean(dur)))

    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc1, np.mean(dur)))
import pandas as pd

acc = pd.DataFrame(data = acc)
acc.to_csv('./test2.csv')