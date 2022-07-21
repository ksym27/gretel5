import numpy as np
import torch

def sampling_mask(observed, starts, targets, n_samples):
    n_observed = len(observed)
    n_indices = min(n_observed, n_samples)
    indices = torch.randperm(n_observed)[0:n_indices].sort().values
    observed = observed[indices, :]
    starts = starts[indices]
    targets = targets[indices]
    return observed, starts, targets


def prepare_data(trajectories, trajectory_idx, graph):
    # 観測ノード列から観測エッジ列を取得
    paths = trajectories._traversed_edges_by_trajectory(trajectory_idx)
    n_steps, mn_edges = paths.size()

    # 観測時間の閉塞データを取得する
    node_times = trajectories.times(trajectory_idx)
    blocked_edges = graph.blockage[:, node_times[:-1]]

    # 時間情報の生成する
    # edge_times = torch.zeros(n_steps, graph.n_edge)
    # for f, p, t in zip(edge_times, paths, node_times):
    #     f[p[p != -1]] = float(t)
    # edge_times = edge_times.t()

    edge_times = (node_times).repeat(graph.n_edge,1)

    return blocked_edges, edge_times
