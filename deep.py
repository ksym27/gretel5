import torch
from future import Future


def blockage(trajectories, trajectory_idx, graph):
    # 観測ノード列から観測エッジ列を取得
    paths = trajectories.traversed_edges_by_trajectory(trajectory_idx)
    n_steps, mn_edges = paths.size()

    # 観測時間の閉塞データを取得する
    node_times = trajectories.times(trajectory_idx)
    blocked_edges = graph.blockage[:, node_times[:-1]]

    # 時間情報の生成する
    # edge_times = torch.zeros(n_steps, graph.n_edge)
    # for f, p, t in zip(edge_times, paths, node_times):
    #     f[p[p != -1]] = float(t)
    # edge_times = edge_times.t()

    # 時間情報の生成する
    edge_times = node_times.repeat(graph.n_edge, 1)

    return blocked_edges, edge_times


def sampling_mask(observed, starts, targets, n_samples):
    n_observed = len(observed)
    n_indices = min(n_observed, n_samples)
    indices = torch.randperm(n_observed)[0:n_indices].sort().values
    observed = observed[indices, :]
    starts = starts[indices]
    targets = targets[indices]
    return observed, starts, targets


def nearest_nodes(nodes, start, graph):
    coords = graph.coords
    min_idx = None
    min_dist = None
    p0 = coords[start]
    for i in nodes:
        p = coords[i]
        dist = torch.linalg.norm(p0 - p)
        if min_idx is None or dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx, min_dist


def update(future: Future, graph, config=None):
    # 予測前
    nodes_with_time = []
    times = future.node_times
    obs = future.observations
    for node, time in zip(obs, times):
        _, topk_nodes = torch.topk(node, 1)
        nodes_with_time.append([topk_nodes[0], time.item(), 2])

    nodes = torch.unique_consecutive(future.nodes)
    pre_node, pre_time_idx, _ = nodes_with_time[-1]

    # 予測
    interval = config.obs_time_intervals
    current_time = 0
    n_steps = 1
    for node in nodes:
        if pre_node != node:
            edge = graph.edge(pre_node, node)
            time = edge[0] / config.agent_speed

            current_time += time
            while (n_steps * interval) <= current_time:
                nodes_with_time.append([node, pre_time_idx + n_steps, 3])
                n_steps += 1
        pre_node = node

    # # 前の時間を埋める
    # node, time, _ = nodes_with_time[0]
    # for i in range(config.sim_start_time_step, time):
    #     nodes_with_time.insert(i, [node, i, 1])
    #
    # # 後ろの時間を埋める
    # node, time, _ = nodes_with_time[-1]
    # for i in range(time+1, config.sim_end_time_step):
    #     nodes_with_time.append([node, i, 4])



    return nodes_with_time
