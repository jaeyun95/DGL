#(1) dgl get a glance

# Copyright is at https://www.dgl.ai
import dgl

def build_karate_club_graph():
    g = dgl.DGLGraph()
    # node 개수를 추가해줌
    g.add_nodes(34)
    # edge 를 정의해줌
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
    # src는 시작점 dst는 도착지점 (예) (1,0)에서 scr는 1, dst는 0
    src, dst = tuple(zip(*edge_list))
	# edge를 추가해줌
    g.add_edges(src, dst)
    # dgl은 directional graph를 다루기 때문에 bi-directional 그래프는 반대쪽으로도 연결해줘야함
    g.add_edges(dst, src)

    return g

G = build_karate_club_graph()
# 그래프 node 출력
print('We have %d nodes.' % G.number_of_nodes())
# 그래프 edge 출력
print('We have %d edges.' % G.number_of_edges())

import networkx as nx
# 시각화를 위해서는 networkx로 바꿔줘야함
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.show()


import torch

# 각 node 별 one-hot vector로 임의의 feature 설정
G.ndata['feat'] = torch.eye(34)

# 2번째 node의 feature 출력
print(G.nodes[2].data['feat'])

# 10, 11번 node의 feature 출력
print(G.nodes[[10, 11]].data['feat'])


import torch.nn as nn
import torch.nn.functional as F

def gcn_message(edges):
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# GCNLayer 정의
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
		# g는 그래프이고, inputs는 입력 node feature
        # node의 feature를 먼저 설정해줌
        g.ndata['h'] = inputs
		# 연결된 모든 edge로 메세지 전달
        g.send(g.edges(), gcn_message)
        # 모든 node에서 전달된 메세지를 통합
        g.recv(g.nodes(), gcn_reduce)
        # 최종 node feature를 뽑음
        h = g.ndata.pop('h')
        return self.linear(h)

# GCN 모델 정의
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
		# 2계층의 gcn
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h
		
net = GCN(34, 5, 2)
inputs = torch.eye(34)

# 라벨이 있는 노드들을 설정해줌 
# 라벨이 존재하는 노드는 0, 33
labeled_nodes = torch.tensor([0, 33])  
# 0, 33의 라벨은 0, 1
labels = torch.tensor([0, 1])  

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    logits = net(G, inputs)
    # 후에 시각화를 위해 값을 저장함
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # 라벨과 비교하여 loss를 계산함
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def draw(i):
	# 각 클래스의 색깔 지정
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
draw(0)  
plt.close()
# 애니메이션 시각화
ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
