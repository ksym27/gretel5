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


def convert_sim(config_filename, input_filename, output_dir):
    config = Config()
    config.load_from_file(config_filename)

    input_dir = os.path.join(config.workspace, config.input_directory)

    graph = Graph.read_from_files_for_deep(
        nodes_filename=os.path.join(input_dir, "nodes.txt"),
        edges_filename=os.path.join(input_dir, "edges.txt"),
        blockage_filename=os.path.join(input_dir, "blockage.txt"),
        shelter_filename=os.path.join(input_dir, "shelter.txt"),
        blockage_time_intervals=config.blockage_time_intervals,
        start_reverse_edges=config.start_reverse_edges
    )

    nx_graph = graph.nx_graphs[0]

    data = []
    lengths = []
    paths = []
    max_path_length = 0
    with open(input_filename) as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            elements = line.replace('\n', '').split(',')
            head_items = elements[0].split(':')
            original_id = head_items[0].split('_')[0]
            start_time = int(float(head_items[1]) * 60)

            records = []
            for j in range(0, len(elements)):
                if j == 0:
                    node = int(head_items[-1])
                    records.append((node, int(1), start_time, i, original_id))
                else:
                    time = start_time + 6 * j
                    node = int(elements[j])
                    records.append((node, int(1), time, i, original_id))
            if len(records) > 1:
                data.extend(records)
                lengths.append((i, len(records)))

                # ルート探索
                for k in range(1, len(records)):
                    source = graph.node_id_map[int(records[k - 1][0])]
                    target = graph.node_id_map[int(records[k][0])]
                    path_nodes = nx.shortest_path(nx_graph, source=source, target=target, weight='weight')
                    n_path_nodes = len(path_nodes)
                    edges = None
                    if n_path_nodes > 1:
                        edges = [nx_graph.get_edge_data(path_nodes[l - 1], path_nodes[l])['id'] for l in
                                 range(1, n_path_nodes)]
                    else:
                        edges = [nx_graph.get_edge_data(path_nodes[0], path_nodes[0])['id']]
                    paths.append(edges)
                    max_path_length = max(max_path_length, len(edges))

    # ファイルに出力する
    observations_filename = os.path.join(output_dir, "observation_6sec.txt")
    lengths_filename = os.path.join(output_dir, "lengths.txt")
    paths_filename = os.path.join(output_dir, "paths.txt")
    blockage_filename = os.path.join(output_dir, "blockage_.txt")

    with open(observations_filename, 'w') as f:
        f.write("%d\t1\n" % len(data))
        f.writelines([("{}\t{}\t{}\t{}\t{}\n".format(i[0], i[1], i[2], i[3], i[4])) for i in data])

    with open(lengths_filename, 'w') as f:
        f.writelines(["%d\t%d\n" % (i[0], i[1]) for i in lengths])

    # write paths
    with open(paths_filename, 'w') as f:
        f.write("{:d}\t{:d}\n".format(len(paths), max_path_length))
        for path in paths:
            path_ = [str(graph.edge_rid_map[i]) for i in path]
            f.write('{}\n'.format('\t'.join(path_)))

    with open(blockage_filename, 'w') as f:
        n1, n2 = graph.blockage.size()
        for i in range(0, n2, 150):
            for j in range(n1):
                condition = graph.blockage[j, i]
                if condition != 0:
                    f.write('{},{}\n'.format(graph.edge_rid_map[j], i))


def convert_blockage(input_filename, blockage_filename):
    # read node features
    lines = []
    with open(input_filename) as f:
        lines = f.readlines()

    with open(blockage_filename, 'w') as f:
        n_columns = len(lines[0].split(','))
        n_head = 3
        f.write("{}\t{}\t{}\n".format(len(lines) - 1, n_head, n_columns - n_head - 1))
        f.writelines(lines[1:])


def convert_obs(config_filename, output_dir):
    config = Config()
    config.load_from_file(config_filename)

    graph, trajectories = load_data2(config)

    n_noloop = graph.n_edge - graph.n_node
    step = 2
    data = []
    ids = []
    n_total = 0
    for i in range(0, len(trajectories)):
        observations = trajectories[i]
        edges = trajectories.traversed_edges_by_trajectory(i)
        times = trajectories.times(i)

        pred_j = 0
        pred_o = None
        candiate_count = 0
        sub = []
        n_observations = len(observations)
        for j in range(n_observations):
            o = torch.argmax(observations[j])
            if j != n_observations - 1:
                if o != pred_o:
                    if candiate_count % step == 0:
                        paths = edges[pred_j:j].flatten()
                        paths = paths[(paths != -1) & (paths < n_noloop)]
                        sub.append((o, times[j], paths))
                        pred_j = j
                    candiate_count += 1
            else:
                if len(sub) == 0 or sub[-1][0] != o:
                    paths = edges[pred_j:j].flatten()
                    paths = paths[(paths != -1) & (paths < n_noloop)]
                    sub.append((o, times[j], paths))
            pred_o = o

        if len(sub) > 1:
            data.append(sub)
            ids.append(i)
            n_total += len(sub)

    observations_filename = os.path.join(output_dir, "observation_6sec_s.txt")
    lengths_filename = os.path.join(output_dir, "lengths_s.txt")
    paths_filename = os.path.join(output_dir, "paths_s.txt")
    size_path = [0, 0]
    with open(observations_filename, 'w') as f1, open(lengths_filename, 'w') as f2:
        f1.write("%d\t1\n" % n_total)
        for i, sub in enumerate(data):
            for j in sub:
                node = graph.node_rid_map[j[0].item()]
                time = j[1].item() * config.obs_time_intervals
                f1.write("%d\t1\t%d\t%d\n" % (node, time, ids[i]))
                n_paths = len(j[2])
                size_path[0] += 1 if n_paths > 0 else 0
                size_path[1] = n_paths if n_paths > size_path[1] else size_path[1]
            f2.write("%d\t%d\n" % (ids[i], len(sub)))

    with open(paths_filename, 'w') as f:
        r, c = size_path
        f.write("{}\t{}\n".format(r, c))
        for i, sub in enumerate(data):
            for j in sub:
                if len(j[2]) > 0:
                    line = "\t".join([str(graph.edge_rid_map[k.item()]) for k in j[2]])
                    f.write("%s\n" % line)

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
        id, node, time, _ = r
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
    # path1 = "/home/owner/dev/gretel3/workspace/deep2/deep_nll.txt"
    # path2 = "/home/owner/dev/gretel3/workspace/deep2/"
    # convert_obs(path1, path2)
    #
    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/prediction_result.txt"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/prediction_result_alloc.txt"
    # # # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/true_result.txt"
    # # # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/true_result_alloc.txt"
    # convert(path1, path2)

    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/result.csv"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/result_rmse.csv"
    # rmse(path1, path2)

    path0 = "/home/owner/dev/gretel3/workspace/deep2/deep_nll.txt"
    path1 = "/home/owner/Desktop/input/traj_pos98_0.1分間隔_5_sep_fin.csv"
    path2 = "/home/owner/Desktop/"
    convert_sim(path0, path1, path2)

    # path0 = "/home/owner/Desktop/input/blockage_new_20220227.csv"
    # path1 = "/home/owner/Desktop/blockage.txt"
    # convert_blockage(path0, path1)
