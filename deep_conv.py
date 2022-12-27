
import csv
import os

def convert(lengths_filename, observations_filename, output_dir):
    lengths = []
    with open(lengths_filename) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            lengths.append(int(row[1]))
    lines = None
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

    lengths_filename = os.path.join(output_dir, "l.txt")
    with open(lengths_filename, 'w') as f:
        for i, lines in enumerate(data):
            f.write("%d\t%d\n" % (i, len(lines)))

    observations_filename = os.path.join(output_dir, "o.txt")
    with open(observations_filename, 'w') as f:
        f.write("%d\t1\n" % total)
        for i, lines in enumerate(data):
            f.writelines(lines)

    return None

if __name__ == "__main__":
    path1 = "/home/owner/dev/gretel3/workspace/deep/observations_6sec.txt"
    path2 = "/home/owner/dev/gretel3/workspace/deep/lengths.txt"
    dir = "/home/owner/dev/gretel3/workspace/deep/"

    convert(path2, path1, dir)
