import csv
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def convert_o(lengths_filename, observations_filename, output_dir):
    lengths = []
    with open(lengths_filename) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            lengths.append(int(row[1]))

    with open(observations_filename) as f:
        f.readline()
        lines = f.readlines()

    total = 0
    pos = 0
    data = []
    for length in lengths:
        sub_lines = lines[pos: pos + length]
        pre_node = None
        new_lines = []
        for line in sub_lines:
            items = line.split()
            node = items[0]
            if pre_node != node:
                new_lines.append(line)
            pre_node = node
        data.append(new_lines)
        total += len(new_lines)
        pos += length

    step_size = 3
    observations_filename = os.path.join(output_dir, "observations_6sec.txt")
    lengths_filename = os.path.join(output_dir, "lengths.txt")
    with open(observations_filename, 'w') as f1, open(lengths_filename, 'w') as f2:
        f1.write("%d\t1\n" % total)
        length = 0
        for i, lines in enumerate(data):
            if i % step_size == 0:
                f.writelines(lines)
                length = step_size + 1
        f.write("%d\t%d\n" % (i, length))

    return None


def convert(filename1, filename2):
    start_time = 0
    end_time = 3600 * 5
    interval = 6

    # データを読み込む
    rows = []
    with open(filename1) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    # 前の時間を埋める
    nodes_time = []
    pre_id = None
    pre_node = None
    pre_time = None
    for r in rows:
        id, node, time, type, goal, _ = r
        if id != pre_id:
            # 後ろを埋める
            if pre_id is not None:
                for i in range(pre_time + interval, end_time, interval):
                    nodes_time.append([pre_id, pre_node, str(i), '1'])
            # 前を埋める
            for i in range(start_time, int(time), interval):
                nodes_time.append([id, node, str(i), '1'])

        nodes_time.append([id, node, time, '0'])
        pre_id, pre_node, pre_time, pre_goal = id, node, int(time), goal

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
    path1 = "/home/owner/dev/gretel3/workspace/deep/lengths.txt.b"
    path2 = "/home/owner/dev/gretel3/workspace/deep/observations_6sec.txt.b"
    path3 = "/home/owner/dev/gretel3/workspace/deep/"
    convert_o(path1, path2, path3)

    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/prediction_result.csv"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/prediction_result_alloc.csv"
    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/true.csv"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/true_alloc.csv"
    # convert(path1, path2)

    # path1 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/result.csv"
    # path2 = "/home/owner/dev/gretel3/workspace/chkpt/deep-nll/result_rmse.csv"
    # rmse(path1, path2)
