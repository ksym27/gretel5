import csv
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import torch

from config import Config
from main import load_data2

def convert_sim(input_filename, observations_filename, lengths_filename):
    # read node features
    data = []
    lengths = []
    total = 0
    with open(input_filename) as f:
        for i, line in enumerate(f.readlines()):
            elements = line.replace('\n', '').split(',')
            records = []
            for j in range(1, len(elements)):
                values = elements[j].split(':')
                time = int(float(values[0]) * 60)
                node = int(values[1])
                records.append("{}\t{}\t{}\t{}\n".format(node, int(1), time, i))
            if len(records) > 1:
                data.append(records)
                lengths.append((i, len(records)))
                total += len(records)

    with open(observations_filename, 'w') as f1, open(lengths_filename, 'w') as f2:
        f1.write("%d\t1\n" % total)
        for i, lines in enumerate(data):
            f1.writelines(lines)
            f2.write("%d\t%d\n" % (lengths[i][0], lengths[i][1]))


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
            if j != n_observations-1:
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
                f1.write("%d\t1\t%d\t%d\n" % (node,  time, i))
                n_paths = len(j[2])
                size_path[0] += 1 if n_paths > 0 else 0
                size_path[1] = n_paths if n_paths > size_path[1] else size_path[1]
            f2.write("%d\t%d\n" % (i, len(sub)))

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
    path1 = "/home/owner/dev/gretel3/workspace/deep1/deep_nll.txt"
    path2 = "/home/owner/dev/gretel3/workspace/deep1/"
    convert_obs(path1, path2)
    #
    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/prediction_result.txt"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/prediction_result_alloc.txt"
    # # # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/true_result.txt"
    # # # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/true_result_alloc.txt"
    # convert(path1, path2)

    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/result.csv"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/result_rmse.csv"
    # rmse(path1, path2)

    # path0 = "/home/owner/Desktop/input/traj_pos65_0.1分間隔_4_sep1.csv"
    # path1 = "/home/owner/Desktop/observation_6sec.txt"
    # path2 = "/home/owner/Desktop/lengths.txt"
    # convert_sim(path0, path1, path2)

    # path0 = "/home/owner/Desktop/input/blockage_new_20220227.csv"
    # path1 = "/home/owner/Desktop/blockage.txt"
    # convert_blockage(path0, path1)
