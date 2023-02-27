import csv
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import networkx as nx

from config import Config
from graph import Graph
from main import load_data2
from utils import numpify


def convert_sim(config_filename, input_filename, block_filename, output_dir):
    config = Config()
    config.load_from_file(config_filename)

    # エッジとノードを読み込み
    input_dir = os.path.join(config.workspace, config.input_directory)
    graph = Graph.read_from_files_for_deep(
        nodes_filename=os.path.join(input_dir, "nodes.txt"),
        edges_filename=os.path.join(input_dir, "edges.txt")
    )

    # nxNetwork生成
    attr_dict = []
    for i, w in enumerate(graph.edges[:, 0]):
        attr_dict.append({'id': i, 'weight': w})
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(zip(numpify(graph.senders), numpify(graph.receivers), attr_dict))

    # データの変換
    data = []
    paths = []
    total_data = 0
    total_path = 0
    max_path_length = 0
    with open(input_filename) as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            elements = line.replace('\n', '').split(',')
            head_items = elements[0].split(':')
            start_time = int(float(head_items[1]) * 60)

            records = []
            for j in range(0, len(elements)):
                if j == 0:
                    node = int(head_items[-1])
                    record = (node, int(1), start_time)
                else:
                    time = start_time + 6 * j
                    node = int(elements[j])
                    record = (node, int(1), time)
                # ノードが存在する場合のみ格納する
                if record[0] in graph.node_id_map:
                    records.append(record)

            if len(records) > 1:
                data.append(records)
                sub_paths = []
                for k in range(1, len(records)):
                    source = graph.node_id_map[int(records[k - 1][0])]
                    target = graph.node_id_map[int(records[k][0])]
                    path_nodes = nx.shortest_path(nx_graph, source=source, target=target, weight='weight')
                    n_path_nodes = len(path_nodes)
                    if n_path_nodes > 1:
                        edges = [nx_graph.get_edge_data(path_nodes[l - 1], path_nodes[l])['id'] for l in
                                 range(1, n_path_nodes)]
                    else:
                        edges = [nx_graph.get_edge_data(path_nodes[0], path_nodes[0])['id']]
                    sub_paths.append(edges)
                    max_path_length = max(max_path_length, len(edges))
                paths.append(sub_paths)

                total_data += len(records)
                total_path += len(sub_paths)

    # Pathを平坦化する
    nodup_edges = set([l3 for l1 in paths for l2 in l1 for l3 in l2])

    # 閉塞データをLoadする
    blockage = [None] * (graph.n_edge - graph.n_node)
    with open(block_filename) as f:
        f.readline()
        for line in f.readlines():
            elements = line.replace('\n', '').split(',')
            ids = [int(elements[1]), int(elements[2])]
            features = list(map(float, elements[3:]))
            for id in ids:
                path_flag = False
                sim_ids = []
                for pair_id in [id, id + config.start_reverse_edges]:
                    if pair_id in graph.edge_id_map:
                        sim_id = graph.edge_id_map[pair_id]
                        sim_ids.append(sim_id)
                        if sim_id in nodup_edges:
                            path_flag = True

                sim_features = [0.] * len(features) if path_flag else features
                for sim_id in sim_ids:
                    blockage[sim_id] = sim_features

    # ファイルに出力する
    observations_filename = os.path.join(output_dir, "observation_6sec.txt")
    lengths_filename = os.path.join(output_dir, "lengths.txt")
    paths_filename = os.path.join(output_dir, "paths.txt")
    blockage_filename = os.path.join(output_dir, "blockage.txt")
    blockage_filename_ = os.path.join(output_dir, "blockage_.txt")


    with open(observations_filename, 'w') as f:
        f.write("%d\t1\n" % total_data)
        for i, e in enumerate(data):
            f.writelines([("{}\t{}\t{}\t{}\n".format(i, j[0], j[1], j[2])) for j in e])

    with open(lengths_filename, 'w') as f:
        for i, e in enumerate(data):
            f.writelines(["%d\t%d\n" % (i, len(e))])

    with open(paths_filename, 'w') as f:
        f.write("{:d}\t{:d}\n".format(total_path, max_path_length))
        for i, path_ in enumerate(paths):
            for edges in path_:
                f.write('{}\t{}\n'.format(i, '\t'.join([str(graph.edge_rid_map[j]) for j in edges])))

    with open(blockage_filename, 'w') as f:
        f.write("{:d}\t{:d}\n".format(len(blockage), len(blockage[0])))
        for i, e in enumerate(blockage):
            f.write('{},{}\n'.format(graph.edge_rid_map[i], ','.join([str(j) for j in e])))

    with open(blockage_filename_, 'w') as f:
        for i in range(len(blockage)):
            f.write('{},{}\n'.format(graph.edge_rid_map[i], blockage[i][0]))


def convert_obs(config_filename, input_dir, output_dir):
    config = Config()
    config.load_from_file(config_filename)

    graph, trajectories = load_data2(config, input_dir)

    n_edges = graph.n_edge - graph.n_node
    step_time = 5
    data = []
    n_total = 0
    for i in range(0, len(trajectories)):
        observations = trajectories[i]
        edges = trajectories.traversed_edges_by_trajectory(i)
        times = trajectories.times(i)

        prev_j = 0
        prev_o = None
        prev_time = -1000
        sub = []
        n_observations = len(observations)
        for j in range(n_observations):
            o = torch.argmax(observations[j])
            if j != n_observations - 1:
                if o != prev_o:
                    if times[j] - prev_time >= step_time:
                        paths = edges[prev_j:j].flatten()
                        paths = paths[(paths != -1) & (paths < n_edges)]
                        sub.append((o, times[j], paths))
                        prev_j = j
                        prev_o = o
                        prev_time = times[j]
            else:
                # 最終ノードはかならず追加する
                if len(sub) == 0 or sub[-1][0] != o:
                    paths = edges[prev_j:j].flatten()
                    paths = paths[(paths != -1) & (paths < n_edges)]
                    sub.append((o, times[j], paths))

        if len(sub) > 1:
            node = sub[-1][0]
            start_time = sub[-1][1] + step_time
            mask = (graph.senders == node.item()) & (graph.receivers == node.item())
            edge = torch.where(mask > 0)[0]
            for j in range(20):
                time = start_time + step_time * j
                if time >= 3600:
                    break
                sub.append((node, time, edge))

            data.append(sub)
            n_total += len(sub)

    observations_filename = os.path.join(output_dir, "observation_6sec_s.txt")
    lengths_filename = os.path.join(output_dir, "lengths_s.txt")
    paths_filename = os.path.join(output_dir, "paths_s.txt")
    paths_filename_ = os.path.join(output_dir, "paths_s_.txt")
    total_paths = 0
    n_max_paths = 0
    with open(observations_filename, 'w') as f1, open(lengths_filename, 'w') as f2:
        f1.write("%d\t1\n" % n_total)
        for i, sub in enumerate(data):
            for j in sub:
                node = graph.node_rid_map[j[0].item()]
                time = j[1].item() * config.obs_time_intervals
                f1.write("%d\t%d\t1\t%d\n" % (i, node, time))
                n_paths = len(j[2])
                total_paths += 1 if n_paths > 0 else 0
                n_max_paths = n_paths if n_paths > n_max_paths else n_max_paths
            f2.write("%d\t%d\n" % (i, len(sub)))

    with open(paths_filename, 'w') as f:
        f.write("{}\t{}\n".format(total_paths, n_max_paths))
        for i, sub in enumerate(data):
            for j in sub:
                if len(j[2]) > 0:
                    line = "\t".join([str(graph.edge_rid_map[k.item()]) for k in j[2]])
                    f.write("%d\t%s\n" % (i, line))

    with open(paths_filename_, 'w') as f:
        for i, sub in enumerate(data):
            for j in sub:
                if len(j[2]) > 0:
                    time = j[1].item() * config.obs_time_intervals
                    for k in j[2]:
                        f.write("%d\t%d\t%d\n" % (i, time, graph.edge_rid_map[k.item()]))

    return None


def convert(filename1, filename2):
    start_time = 0
    end_time = 3600 * 5
    interval = 6

    # データを読み込む
    rows = []
    with open(filename1) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            rows.append(row)

    # 前の時間を埋める
    nodes_time = []
    pre_id = None
    pre_node = None
    pre_time = None
    for r in rows:
        id, node, time, _, _ = r
        if id != pre_id:
            # 後ろを埋める
            if pre_id is not None:
                for i in range(pre_time + interval, end_time, interval):
                    nodes_time.append([pre_id, pre_node, str(i), '1'])
            # 前を埋める
            for i in range(start_time, int(time), interval):
                nodes_time.append([id, node, str(i), '1'])
                pre_time = i

        for i in range(pre_time + interval, int(time) + interval, interval):
            nodes_time.append([id, node, str(i), '0'])

        pre_id, pre_node, pre_time = id, node, int(time)

    # データの書き込み
    with open(filename2, 'w') as f:
        for i in nodes_time:
            line = ','.join(i)
            f.write('%s\n' % (line))

    return None


def rmse(filename1, filename2):
    interval = 6
    start = 0
    end = 3600 * 5
    df = pd.read_csv(filename1, header=0)
    for time in range(start, end, interval):
        sub = df[df['time'] == time]
        rmse = np.sqrt(mean_squared_error(sub['true'], sub['pred']))
        print(time, rmse)

    return None


if __name__ == "__main__":

    no = 4
    if False:
        path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll%d/prediction_result.txt" % no
        path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll%d/prediction_result_alloc.txt" % no
        # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll%d/true_result.txt" % no
        # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll%d/true_result_alloc.txt" % no
        convert(path1, path2)

    if True:
        path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll%d/result.csv" % no
        path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll%d/result_rmse.csv" % no
        rmse(path1, path2)

    path0 = "/home/owner/dev/gretel3/workspace/deep%d/deep_nll.txt" % no
    path1 = "/home/owner/Desktop/input/traj_pos98_0.1分間隔_5_sep_fin.csv"
    path2 = "/home/owner/Desktop/input/blockage_new_20220227.csv"
    path3 = "/home/owner/Desktop/input/r/"
    path4 = "/home/owner/Desktop/input/r/"

    if False:
        convert_sim(path0, path1, path2, path3)

    if False:
        convert_obs(path0, path3, path4)
