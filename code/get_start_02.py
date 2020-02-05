#(2) dglGraph and node/edge features

# Copyright is at https://www.dgl.ai

import networkx as nx
import dgl

# networkx graph 생성
# petersen graph : 10개의 꼭지점, 15개의 선이 있는 무방향 그래프
g_nx = nx.petersen_graph()
# dgl graph 생성(networkx graph를 변환)
g_dgl = dgl.DGLGraph(g_nx)

# 그리기
import matplotlib.pyplot as plt
plt.subplot(121)
nx.draw(g_nx, with_labels=True)
plt.subplot(122)
nx.draw(g_dgl.to_networkx(), with_labels=True)

plt.show()

import dgl
import torch as th

# dgl graph 생성
g = dgl.DGLGraph()
# node 추가
g.add_nodes(10)
# 1-4 edge 추가(하나씩 추가하는 법)
for i in range(1, 5):
    g.add_edge(i, 0)
# 5-7 edge 추가(리스트로 추가하는 법)
src = list(range(5, 8)); dst = [0]*3
g.add_edges(src, dst)
# 8-9 edge 추가(텐서로 추가하는 법)
src = th.tensor([8, 9]); dst = th.tensor([0, 0])
g.add_edges(src, dst)

# 싹 지우고 새로 한번에 만들기
g.clear(); g.add_nodes(10)
src = th.tensor(list(range(1, 10)));
g.add_edges(src, 0)

#시각화
import networkx as nx
import matplotlib.pyplot as plt
nx.draw(g.to_networkx(), with_labels=True)
plt.show()

import dgl
import torch as th

# node feature 정의
x = th.randn(10, 3)
# node feature 등록
g.ndata['x'] = x

# node feature 출력
print(g.ndata['x'] == g.nodes[:].data['x'])

# node feature 등록
g.nodes[0].data['x'] = th.zeros(1, 3)
g.nodes[[0, 1, 2]].data['x'] = th.zeros(3, 3)
g.nodes[th.tensor([0, 1, 2])].data['x'] = th.zeros(3, 3)

# edge feature 등록
g.edata['w'] = th.randn(9, 2)

# edge feature 등록
g.edges[1].data['w'] = th.randn(1, 2)
g.edges[[0, 1, 2]].data['w'] = th.zeros(3, 2)
g.edges[th.tensor([0, 1, 2])].data['w'] = th.zeros(3, 2)

# edge 1 -> 0 feature 등록
g.edges[1, 0].data['w'] = th.ones(1, 2)          
# edges [1, 2, 3] -> 0 feature 등록
g.edges[[1, 2, 3], [0, 0, 0]].data['w'] = th.ones(3, 2) 

print(g.node_attr_schemes())
g.ndata['x'] = th.zeros((10, 4))
print(g.node_attr_schemes())


# 기존에 사용했던 feature data 지우기
# node feature 삭제
g.ndata.pop('x')
# edge feature 삭제
g.edata.pop('w')

# multigraph = True 설정을 해주면 생성 가능
g_multi = dgl.DGLGraph(multigraph=True)
# node 추가
g_multi.add_nodes(10)
# node feature 등록
g_multi.ndata['x'] = th.randn(10, 2)

# edge 추가
g_multi.add_edges(list(range(1, 10)), 0)
# edge 등록
g_multi.add_edge(1, 0)

# edge feature 등록
g_multi.edata['w'] = th.randn(10, 2)
# edge feature 등록
g_multi.edges[1].data['w'] = th.zeros(1, 2)
print(g_multi.edges())

# 이전 edge feature 출력
print('before : ',g_multi.edata['w'])
# edge 가 (1, 0) 인 edge들의 id 저장
eid_10 = g_multi.edge_id(1, 0)
# edge 가 (1, 0) 인 edge들의 id 출력
print('multi edge id : ',eid_10)
# edge 가 (1, 0) 인 edge들의 feature를 1로 변경
g_multi.edges[eid_10].data['w'] = th.ones(len(eid_10), 2)
# 이후 edge feature 출력
print('after : ',g_multi.edata['w'])
