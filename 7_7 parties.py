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
import crypten
import crypten.mpc as mpc
from dgl.data import citation_graph as citegrh
from funcs import load_cora_data, Net1, Net2, Net3, evaluate

crypten.init()
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''#########################加载数据，构建子图#########################################'''
start = time.time()

g, features, labels, train_mask, test_mask = load_cora_data()
g.ndata['h'] = features
node_list = list(range(2708))  # 点集
src, dst = g.all_edges()
src = src.detach().numpy()
dst = dst.detach().numpy()
edge_list = list(zip(src, dst))  # 边集

# L = list(range(2708))  # 点集
# list1 = random.sample(range(1, 2708), 50)

list1 = [0, 6, 500]
list2 = [5, 78, 108]
list3 = [388, 788]
list4 = [56, 1006]
list5 = [566, 1008]
list6 = [565, 1567]
list7 = list(set(node_list) - set(list1) - set(list2) - set(list3) - set(list4) - set(list5) - set(list6))

g1 = g.subgraph(list1)  # 构建子图架构
g1.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g2 = g.subgraph(list2)  # 构建子图架构
g2.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g3 = g.subgraph(list3)  # 构建子图架构
g3.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g4 = g.subgraph(list4)  # 构建子图架构
g4.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g5 = g.subgraph(list5)  # 构建子图架构
g5.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g6 = g.subgraph(list6)  # 构建子图架构
g6.copy_from_parent()  # 上面只是拓扑结构，现添加特征
g7 = g.subgraph(list7)  # 构建子图架构
g7.copy_from_parent()  # 上面只是拓扑结构，现添加特征

train_mask1 = train_mask[list1]
train_mask2 = train_mask[list2]
train_mask3 = train_mask[list3]
train_mask4 = train_mask[list4]
train_mask5 = train_mask[list5]
train_mask6 = train_mask[list6]
train_mask7 = train_mask[list7]
labels1 = labels[list1]
labels2 = labels[list2]
labels3 = labels[list3]
labels4 = labels[list4]
labels5 = labels[list5]
labels6 = labels[list6]
labels7 = labels[list7]
features1 = features[list1]
features2 = features[list2]
features3 = features[list3]
features4 = features[list4]
features5 = features[list5]
features6 = features[list6]
features7 = features[list7]

test_mask1 = test_mask[list1]
test_mask2 = test_mask[list2]
test_mask3 = test_mask[list3]
test_mask4 = test_mask[list4]
test_mask5 = test_mask[list5]
test_mask6 = test_mask[list6]
test_mask7 = test_mask[list7]


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
        b = crypten.cryptensor(a)  # 加密上传
        common_nodes[i] = b  # 为对应的点 添加值


def upload_en1(g_pe, list_up):  # 第一轮以后，上传到字典里的是h的特征
    for i in list_up:
        a = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h']  # 对应到子图上的节点
        a = a[0]
        b = crypten.cryptensor(a)
        common_nodes[i] = b  # 为对应的点 添加值


def download_en(g_pe, node_dic, par):  # 从server的字典里面拿到加密的数据,解密并更新子图
    a = list(node_dic.keys())  # 该子图中需要下载更新的节点
    aa, bb = par.size()  # 16,1433
    for i in a:
        b = list(node_dic[i])  # 子图中和i节点相连的其他节点
        s = g_pe.ndata['h'][0].size()  # 该图特征维度，维度随着层数而变换
        sum_en = torch.zeros(s).to(device)
        for j in b:  # 对每一个邻居
            sum_en_sub = torch.zeros(aa).to(device)  # 16个数，每个点16个特征
            sum_en_sub = crypten.cryptensor(sum_en_sub)
            for m in range(aa):
                sum_en_sub[m] = sum(par[m] * common_nodes[j])
            sum_en_sub = sum_en_sub.get_plain_text()  # 解密

            sum_en = sum_en + sum_en_sub.to(device)  # 把server中与i点相连的加密特征 相加
        d = sum_en
        g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] = g_pe.nodes[g_pe.map_to_subgraph_nid(i)].data['h'] + d
    return g_pe


'''######################### 定义server #########################################'''
dic1 = get_download_nodes(g1)  # g1需要连接的其他图上的节点，序号为原始图中点的序号
dic2 = get_download_nodes(g2)  # g2需要连接的其他图上的节点，序号为原始图中点的序号
dic3 = get_download_nodes(g3)
dic4 = get_download_nodes(g4)  # g2需要连接的其他图上的节点，序号为原始图中点的序号
dic5 = get_download_nodes(g5)
dic6 = get_download_nodes(g6)  # g2需要连接的其他图上的节点，序号为原始图中点的序号
dic7 = get_download_nodes(g7)
#
list_common = list(dic1.values()) + list(dic2.values()) + list(dic3.values()) + list(dic4.values()) + list(dic5.values()) + list(dic6.values()) + list(dic7.values())  # 总的需要更新的点，即server上需要保存的点，序号为原始图中点的序号
list_common = _flatten(list_common)
list_common = list(set(list_common))  # 去除重复点
common_nodes = {}  # 将需要更新的点生成字典放到 server
for i in list_common:
    common_nodes.setdefault(i, [])

list_up1 = get_upload_nodes(g1)  # g1上传到server上的点
list_up2 = get_upload_nodes(g2)
list_up3 = get_upload_nodes(g3)
list_up4 = get_upload_nodes(g4)
list_up5 = get_upload_nodes(g5)
list_up6 = get_upload_nodes(g6)
list_up7 = get_upload_nodes(g7)
'''########################定义密钥生成，加密，和解密函数################################'''
upload_en(g1, list_up1)  # 加密上传
upload_en(g2, list_up2)
upload_en(g3, list_up3)
upload_en(g4, list_up4)
upload_en(g5, list_up5)
upload_en(g6, list_up6)
upload_en(g7, list_up7)
'''####################################定义网络结构#####################################'''
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

net_A1 = Net1()  # A方第一层的线性变换，需要拿到参数上传
net_A2 = Net2()  # A方第一层后的激活层
net_A3 = Net3()  # A方第二层

net_B1 = Net1()  # B方第一层
net_B2 = Net2()  # B方第二层
net_B3 = Net3()

net_C1 = Net1()  # B方第一层
net_C2 = Net2()  # B方第二层
net_C3 = Net3()

net_D1 = Net1()  # B方第一层
net_D2 = Net2()  # B方第二层
net_D3 = Net3()

net_E1 = Net1()  # B方第一层
net_E2 = Net2()  # B方第二层
net_E3 = Net3()

net_F1 = Net1()  # B方第一层
net_F2 = Net2()  # B方第二层
net_F3 = Net3()

net_G1 = Net1()  # B方第一层
net_G2 = Net2()  # B方第二层
net_G3 = Net3()

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

optimizer3 = torch.optim.Adam([
    {'params': net_C1.parameters(), 'lr': 1e-1},
    {'params': net_C2.parameters(), 'lr': 1e-1},
    {'params': net_C3.parameters(), 'lr': 1e-1}
])
optimizer4 = torch.optim.Adam([
    {'params': net_D1.parameters(), 'lr': 1e-1},
    {'params': net_D2.parameters(), 'lr': 1e-1},
    {'params': net_D3.parameters(), 'lr': 1e-1}
])
optimizer5 = torch.optim.Adam([
    {'params': net_E1.parameters(), 'lr': 1e-1},
    {'params': net_E2.parameters(), 'lr': 1e-1},
    {'params': net_E3.parameters(), 'lr': 1e-1}
])
optimizer6 = torch.optim.Adam([
    {'params': net_F1.parameters(), 'lr': 1e-1},
    {'params': net_F2.parameters(), 'lr': 1e-1},
    {'params': net_F3.parameters(), 'lr': 1e-1}
])
optimizer7 = torch.optim.Adam([
    {'params': net_G1.parameters(), 'lr': 1e-1},
    {'params': net_G2.parameters(), 'lr': 1e-1},
    {'params': net_G3.parameters(), 'lr': 1e-1}
])

dur = []
all_logits1 = []
all_logits2 = []
all_logits3 = []
all_logits4 = []
all_logits5 = []
all_logits6 = []
all_logits7 = []

end = time.time()
print("前期运行时间:%.2f秒" % (end - start))

for epoch in range(500):
    print(epoch)

    t0 = time.time()

    net_A1.train()
    net_B1.train()
    net_C1.train()
    net_D1.train()
    net_E1.train()
    net_F1.train()
    net_G1.train()
    net_A2.train()
    net_B2.train()
    net_C2.train()
    net_D2.train()
    net_E2.train()
    net_F2.train()
    net_G2.train()
    net_A3.train()
    net_B3.train()
    net_C3.train()
    net_D3.train()
    net_E3.train()
    net_F3.train()
    net_G3.train()

    g1.update_all(gcn_msg, gcn_reduce)  # A聚合,针对 h
    logits_A1 = net_A1(g1, g1.ndata['h'])  # A的第一层,线性变换
    for parameters in net_A1.parameters():  # 线性变换的参数
        par1 = parameters
    par1 = crypten.cryptensor(par1)  # 参数加密上传
    download_en(g1, dic1, par1)  # 下载, 针对 h，是将h加上服务器上的加密值乘以参数再解密
    logits_A1 = net_A2(g1, logits_A1)  # Relu

    g2.update_all(gcn_msg, gcn_reduce)  # B聚合,针对 h
    logits_B1 = net_B1(g2, g2.ndata['h'])  # B的第一层
    for parameters in net_B1.parameters():  # 线性变换的参数
        par2 = parameters
    par2 = crypten.cryptensor(par2)  # 参数加密上传
    download_en(g2, dic2, par2)  # 下载, 针对 h，是将h加上服务器上的d
    logits_B1 = net_B2(g2, logits_B1)  # relu

    g3.update_all(gcn_msg, gcn_reduce)  # C
    logits_C1 = net_C1(g3, g3.ndata['h'])
    for parameters in net_C1.parameters():
        par3 = parameters
    par3 = crypten.cryptensor(par3)
    download_en(g3, dic3, par3)
    logits_C1 = net_C2(g3, logits_C1)

    g4.update_all(gcn_msg, gcn_reduce)  # D
    logits_D1 = net_D1(g4, g4.ndata['h'])
    for parameters in net_D1.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g4, dic4, par)
    logits_D1 = net_D2(g4, logits_D1)

    g5.update_all(gcn_msg, gcn_reduce)  # E
    logits_E1 = net_E1(g5, g5.ndata['h'])
    for parameters in net_E1.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g5, dic5, par)
    logits_E1 = net_E2(g5, logits_E1)

    g6.update_all(gcn_msg, gcn_reduce)  # F
    logits_F1 = net_F1(g6, g6.ndata['h'])
    for parameters in net_F1.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g6, dic6, par)
    logits_F1 = net_F2(g6, logits_F1)

    g7.update_all(gcn_msg, gcn_reduce)  # G
    logits_G1 = net_G1(g7, g7.ndata['h'])
    for parameters in net_G1.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g7, dic7, par)
    logits_G1 = net_G2(g7, logits_G1)
    ###
    upload_en1(g1, list_up1)  # 第一个图同步上传，传到特征h
    upload_en1(g2, list_up2)  # 第二个图同步上传
    upload_en1(g3, list_up3)  #
    upload_en1(g4, list_up4)  #
    upload_en1(g5, list_up5)  #
    upload_en1(g6, list_up6)  #
    upload_en1(g7, list_up7)  #
    # print('第一层后%s' % common_nodes[0].size())

    #############44444444444444  第二层 444444444 ##################

    g1.update_all(gcn_msg, gcn_reduce)  # A聚合,针对 h
    logits1 = net_A3(g1, logits_A1)  # A的第二层
    for parameters in net_A3.parameters():  # 线性变换的参数
        par4 = parameters
    par4 = crypten.cryptensor(par4)  # 参数加密上传
    download_en(g1, dic1, par4)  # 下载, 针对 h，是将h加上服务器上的d
    # print(g1.ndata['h'].size())

    g2.update_all(gcn_msg, gcn_reduce)  # B聚合,针对 h
    logits2 = net_B3(g2, logits_B1)  # B的第二层
    for parameters in net_B3.parameters():  # 线性变换的参数
        par5 = parameters
    par5 = crypten.cryptensor(par5)  # 参数加密上传
    download_en(g2, dic2, par5)  # 下载, 针对 h，是将h加上服务器上的d

    g3.update_all(gcn_msg, gcn_reduce)  # C
    logits3 = net_C3(g3, logits_C1)
    for parameters in net_C3.parameters():
        par6 = parameters
    par6 = crypten.cryptensor(par6)
    download_en(g3, dic3, par6)

    g4.update_all(gcn_msg, gcn_reduce)  # D
    logits4 = net_D3(g4, logits_D1)
    for parameters in net_D3.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g4, dic4, par)

    g5.update_all(gcn_msg, gcn_reduce)  # E
    logits5 = net_E3(g5, logits_E1)
    for parameters in net_E3.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g5, dic5, par)

    g6.update_all(gcn_msg, gcn_reduce)  # F
    logits6 = net_F3(g6, logits_F1)
    for parameters in net_F3.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g6, dic6, par)

    g7.update_all(gcn_msg, gcn_reduce)  # G
    logits7 = net_G3(g7, logits_G1)
    for parameters in net_G3.parameters():
        par = parameters
    par = crypten.cryptensor(par)
    download_en(g7, dic7, par)


    all_logits1.append(logits1.detach())  # 记录
    logp1 = F.log_softmax(logits1, 1)
    loss1 = F.nll_loss(logp1[train_mask1], labels1[train_mask1])

    all_logits2.append(logits2.detach())  # 记录
    logp2 = F.log_softmax(logits2, 1)
    loss2 = F.nll_loss(logp2[train_mask2], labels2[train_mask2])

    all_logits3.append(logits3.detach())  # 记录
    logp3 = F.log_softmax(logits3, 1)
    loss3 = F.nll_loss(logp3[train_mask3], labels3[train_mask3])

    all_logits4.append(logits4.detach())  # 记录
    logp4 = F.log_softmax(logits4, 1)
    loss4 = F.nll_loss(logp4[train_mask4], labels4[train_mask4])

    all_logits5.append(logits5.detach())  # 记录
    logp5 = F.log_softmax(logits5, 1)
    loss5 = F.nll_loss(logp5[train_mask5], labels5[train_mask5])

    all_logits6.append(logits6.detach())  # 记录
    logp6 = F.log_softmax(logits6, 1)
    loss6 = F.nll_loss(logp6[train_mask6], labels6[train_mask6])

    all_logits7.append(logits7.detach())  # 记录
    logp7 = F.log_softmax(logits7, 1)
    loss7 = F.nll_loss(logp7[train_mask7], labels7[train_mask7])

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    optimizer5.zero_grad()
    optimizer6.zero_grad()
    optimizer7.zero_grad()
    loss1.backward(retain_graph=True)
    loss2.backward()
    loss3.backward()
    loss4.backward()
    loss5.backward()
    loss6.backward()
    loss7.backward()
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    optimizer5.step()
    optimizer6.step()
    optimizer7.step()

    dur.append(time.time() - t0)

    p1 = net_A1.state_dict()  # 拿到net1参数字典
    p2 = net_B1.state_dict()  # 拿到net2参数字典
    p3 = net_C1.state_dict()  # 拿到net2参数字典
    p4 = net_D1.state_dict()  # 拿到net2参数字典
    p5 = net_E1.state_dict()  # 拿到net2参数字典
    p6 = net_F1.state_dict()  # 拿到net2参数字典
    p7 = net_G1.state_dict()  # 拿到net2参数字典
    for key, value in p7.items():  # p1等于两个字典平均
        p1[key] = p1[key]*(len(list1)/2708) + p2[key]*(len(list2)/2708) + p3[key]*(len(list3)/2708) + p4[key]*(len(list4)/2708) + p5[key]*(len(list5)/2708) + p6[key]*(len(list6)/2708) + value*(len(list7)/2708)

    net_A1.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B1.load_state_dict(p1)  # net2的参数更新为平均参数
    net_C1.load_state_dict(p1)  # net2的参数更新为平均参数
    net_D1.load_state_dict(p1)  # net2的参数更新为平均参数
    net_E1.load_state_dict(p1)  # net2的参数更新为平均参数
    net_F1.load_state_dict(p1)  # net2的参数更新为平均参数
    net_G1.load_state_dict(p1)  # net2的参数更新为平均参数

    p1 = net_A2.state_dict()  # 拿到net1参数字典
    p2 = net_B2.state_dict()  # 拿到net2参数字典
    p3 = net_C2.state_dict()  # 拿到net2参数字典
    p4 = net_D2.state_dict()  # 拿到net2参数字典
    p5 = net_E2.state_dict()  # 拿到net2参数字典
    p6 = net_F2.state_dict()  # 拿到net2参数字典
    p7 = net_G2.state_dict()  # 拿到net2参数字典
    for key, value in p7.items():  # p1等于两个字典平均
        p1[key] = p1[key] * (len(list1) / 2708) + p2[key] * (len(list2) / 2708) + p3[key] * (len(list3) / 2708) + p4[
            key] * (len(list4) / 2708) + p5[key] * (len(list5) / 2708) + p6[key] * (len(list6) / 2708) + value * (
                              len(list7) / 2708)

    net_A2.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B2.load_state_dict(p1)  # net2的参数更新为平均参数
    net_C2.load_state_dict(p1)  # net2的参数更新为平均参数
    net_D2.load_state_dict(p1)  # net2的参数更新为平均参数
    net_E2.load_state_dict(p1)  # net2的参数更新为平均参数
    net_F2.load_state_dict(p1)  # net2的参数更新为平均参数
    net_G2.load_state_dict(p1)  # net2的参数更新为平均参数

    p1 = net_A3.state_dict()  # 拿到net1参数字典
    p2 = net_B3.state_dict()  # 拿到net2参数字典
    p3 = net_C3.state_dict()  # 拿到net2参数字典
    p4 = net_D3.state_dict()  # 拿到net2参数字典
    p5 = net_E3.state_dict()  # 拿到net2参数字典
    p6 = net_F3.state_dict()  # 拿到net2参数字典
    p7 = net_G3.state_dict()  # 拿到net2参数字典
    for key, value in p7.items():  # p1等于两个字典平均
        p1[key] = p1[key] * (len(list1) / 2708) + p2[key] * (len(list2) / 2708) + p3[key] * (len(list3) / 2708) + p4[
            key] * (len(list4) / 2708) + p5[key] * (len(list5) / 2708) + p6[key] * (len(list6) / 2708) + value * (
                              len(list7) / 2708)


    net_A3.load_state_dict(p1)  # net1的参数更新为平均参数
    net_B3.load_state_dict(p1)  # net2的参数更新为平均参数
    net_C3.load_state_dict(p1)  # net2的参数更新为平均参数
    net_D3.load_state_dict(p1)  # net2的参数更新为平均参数
    net_E3.load_state_dict(p1)  # net2的参数更新为平均参数
    net_F3.load_state_dict(p1)  # net2的参数更新为平均参数
    net_G3.load_state_dict(p1)  # net2的参数更新为平均参数

    loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7) / 7

    g1.ndata['h'] = features1
    g2.ndata['h'] = features2
    g3.ndata['h'] = features3
    g4.ndata['h'] = features4
    g5.ndata['h'] = features5
    g6.ndata['h'] = features6
    g7.ndata['h'] = features7

    upload_en(g1, list_up1)  # 重新给服务器上的字典赋值初始特征
    upload_en(g2, list_up2)
    upload_en(g3, list_up3)
    upload_en(g4, list_up4)
    upload_en(g5, list_up5)
    upload_en(g6, list_up6)
    upload_en(g7, list_up7)

    acc = evaluate(net_A1, net_A2, net_A3, g, features, labels, test_mask)

    # print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
    #     epoch, loss.item(), np.mean(dur)))

    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc, np.mean(dur)))
