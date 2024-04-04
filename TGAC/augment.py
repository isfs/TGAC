
import torch
from torch_geometric.utils import degree, to_undirected, to_networkx
import numpy as np
from torch_scatter import scatter
import networkx as nx
from utils.data_processing import Data

# degree
def degree_drop_weights(edge_index, t):
    edge_index = torch.from_numpy(np.array(edge_index))
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    deg_col = 10*deg_col + (t+1)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

def drop_edge_weighted(src, dst, ts, y, edge_idx, edge_weights, p: float, threshold: float = 1):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool).cpu()
    
    return src[sel_mask], dst[sel_mask], ts[sel_mask], y[sel_mask], edge_idx[sel_mask]

# pagerank
def pr(edge_index):
    edge_index = torch.from_numpy(np.array(edge_index))
    src_list = edge_index[0]
    dst_list = edge_index[1]
    num_nodes = max(max(src_list), max(dst_list)) + 1
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(len(src_list)):
        src, dst = src_list[i], dst_list[i]
        adj[src, dst] += 1

    # computer pagerank
    alpha = 0.85
    num_iter = 100
    pr = torch.ones(num_nodes) / num_nodes
    for i in range(num_iter):
        new_pr = (1 - alpha) / num_nodes + alpha * torch.matmul(adj, pr)
        if torch.allclose(new_pr, pr, rtol=1e-6, atol=1e-6):
            break
        pr = new_pr
    return pr

# pagerank
def pr_drop_weights(edge_index, t, aggr: str = 'sink', k: int = 10):
    edge_index = torch.from_numpy(np.array(edge_index))
    
    pv = compute_pr(edge_index, k=k)
    # pv = pr(edge_index)
    pv_row = (10*pv[edge_index[0]]+(t+1)).to(torch.float32)
    pv_col = (10*pv[edge_index[1]]+(t+1)).to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    tol = 1e-6
    last_v = x
    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1 - damp) * x + damp * agg_msg
        if torch.norm(x - last_v) < tol:
            break
        last_v = x
    return x

def sp_drop_weights(edge_index, t, temporal_bias=1e-6, spatial_bias=1.0):
    time_delta = t[-1]+2 - t 
    temperal_weight = np.exp(- temporal_bias * time_delta)
    temperal_weight = temperal_weight / temperal_weight.sum()
    
    ngh_degs = np.zeros(torch.from_numpy(np.array(edge_index)).max().item() + 1) 
    degs = []
    for i in range(len(edge_index[0])):
        ngh_degs[edge_index[0][i]] += 1
        ngh_degs[edge_index[1][i]] += 1 
        degs.append((ngh_degs[edge_index[0][i]]+ngh_degs[edge_index[1][i]]))
    spatial_weight = np.exp([- spatial_bias / (i + 0.01) for i in degs])
    spatial_weight = spatial_weight / spatial_weight.sum()
    
    x = (temperal_weight + spatial_weight) / 2.0
    x = torch.from_numpy(x)
    s_col = torch.log(x)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    
    return weights

def random_drop_weights(edge_index):
    random_weights = np.random.random(size=len(edge_index[0]))
    random_weights = torch.from_numpy(random_weights)
    return random_weights


# evc
def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    src, dst = data.sources, data.destinations
    s_row, s_col = 10*s[src]+data.timestamps+1, 10*s[dst]+data.timestamps+1
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())

def eigenvector_centrality(data):
    edge_index = torch.tensor([data.sources, data.destinations])
    graph = nx.Graph()
    graph.add_edges_from(edge_index.t().tolist())

    x = nx.eigenvector_centrality(graph, max_iter=2000, tol=1.0e-5)

    l = max(np.max(data.sources), np.max(data.destinations))+1
    ec = [0] * l
    for key, value in x.items():
        ec[key] = value
    return torch.tensor(ec, dtype=torch.float32)


def prune(data, edge_weights, c = 0.05):
    sources = data.sources
    destinations = data.destinations
    timestamps = data.timestamps
    labels = data.labels
    edge_idxs = data.edge_idxs
    num_edges = len(sources)
    edge_weights = edge_weights / edge_weights.mean()
    k = int(num_edges * (1 - c))
    _, indices = torch.topk(edge_weights, k)
    sources = [sources[i] for i in indices]
    destinations = [destinations[i] for i in indices]
    timestamps = [timestamps[i] for i in indices]
    labels = [labels[i] for i in indices]
    edge_idxs = [edge_idxs[i] for i in indices]
    
    
    prune_data = Data(np.array(sources), np.array(destinations), np.array(timestamps), np.array(edge_idxs), np.array(labels))

    return prune_data
