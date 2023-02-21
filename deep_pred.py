import argparse
import os
import sys

import torch
import csv

from typing import Callable
from config import Config
from future import Future
from metrics import Evaluator
from main import load_data, create_optimizer, create_model, evaluate
from tqdm import tqdm


def pre_process(config):
    # load data
    graph, trajectories, pairwise_node_features, _ = load_data(config)
    train_trajectories, valid_trajectories, test_trajectories = trajectories

    graph = graph.to(config.device)

    if not config.name:
        experiment_name = input("Give a name to this experiment? ").strip()
        config.name = experiment_name or config.date

    print(f'==== START "{config.name}" ====')

    torch.manual_seed(config.seed)

    chkpt_dir = os.path.join(config.workspace, config.checkpoint_directory, config.name)
    if config.enable_checkpointing:
        os.makedirs(chkpt_dir, exist_ok=True)
        print(f"Checkpoints will be saved in [{chkpt_dir}]")

    d_node = graph.nodes.shape[1] if graph.nodes is not None else 0
    d_edge = graph.edges.shape[1] if graph.edges is not None else 0
    print(f"Number of node features {d_node}. Number of edge features {d_edge}")

    model = create_model(graph, pairwise_node_features, config)
    model = model.to(config.device)

    optimizer = create_optimizer(model.parameters(), config)

    filename = os.path.join(chkpt_dir, config.checkpoint_file_name)
    checkpoint_data = torch.load(filename)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    graph = graph.add_self_loops(
        degree_zero_only=config.self_loop_deadend_only, edge_value=config.self_loop_weight
    )

    if config.rw_non_backtracking:
        print("Computing non backtracking graph...", end=" ")
        sys.stdout.flush()
        graph.compute_non_backtracking_edges()
        print("Done")

    # マスクを出力
    chkpt_dir = os.path.join(config.workspace, config.checkpoint_directory, config.name)
    mask_filename = os.path.join(chkpt_dir, 'mask_test.csv')
    with open(mask_filename, 'w') as f:
        for i, m in enumerate(test_trajectories._mask):
            f.write('%d,%d\n' % (i, m.item()))

    chkpt_dir = os.path.join(config.workspace, config.checkpoint_directory, config.name)
    mask_filename = os.path.join(chkpt_dir, 'mask_train.csv')
    with open(mask_filename, 'w') as f:
        for i, m in enumerate(train_trajectories._mask):
            f.write('%d,%d\n' % (i, m.item()))

    return graph, test_trajectories, model


def evaluate(
        model,
        graph,
        trajectories,
        trajectory_idx,
        evaluator_creator: Callable[[], Evaluator],
        future,
        end_time
) -> Evaluator:
    model.eval()
    evaluator = evaluator_creator()
    return evaluator.predict(model, graph, trajectories, trajectory_idx, future, end_time)


def load_prediction(config, dir_name, graph):
    observations_fn = os.path.join(dir_name, "pred_observations.txt")
    observation_time_fn = os.path.join(dir_name, "pred_observation_times.txt")
    observation_steps_fn = os.path.join(dir_name, "pred_observation_steps.txt")
    predicted_nodes_fn = os.path.join(dir_name, "pred_nodes.txt")
    predicted_times_fn = os.path.join(dir_name, "pred_times.txt")
    condition_fn = os.path.join(dir_name, "pred_condition.txt")

    future = Future(1)

    with open(observations_fn, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [list(map(int, r)) for r in reader]
        future.observations = [torch.tensor(l) for l in lines]

    with open(observation_time_fn, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [list(map(int, r)) for r in reader]
        future.observation_times = [torch.tensor(l) for l in lines]

    with open(observation_steps_fn, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [list(map(int, r)) for r in reader]
        future.observation_steps = [torch.tensor(l) for l in lines]

    with open(predicted_nodes_fn, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [list(map(int, r)) for r in reader]
        future.predicted_nodes = [torch.tensor(l) for l in lines]

    with open(predicted_times_fn, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [list(map(int, r)) for r in reader]
        future.predicted_times = [torch.tensor(l) for l in lines]

    with open(condition_fn, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        future.condition = [int(row[0]) for row in reader]

    return future


def write_prediction(config, future, dir_name, graph):
    observations_fn = os.path.join(dir_name, "pred_observations.txt")
    observation_time_fn = os.path.join(dir_name, "pred_observation_times.txt")
    observation_steps_fn = os.path.join(dir_name, "pred_observation_steps.txt")
    predicted_nodes_fn = os.path.join(dir_name, "pred_nodes.txt")
    predicted_times_fn = os.path.join(dir_name, "pred_times.txt")
    condition_fn = os.path.join(dir_name, "pred_condition.txt")
    trajectory_fn = os.path.join(dir_name, "prediction_result.txt")

    with open(observations_fn, 'w') as f:
        for i, l in enumerate(future.observations):
            line = "\t".join(str(torch.argmax(n).item()) for n in l) + "\n"
            f.write(line)

    with open(observation_time_fn, "w") as f:
        for i, l in enumerate(future.observation_times):
            line = "\t".join(str(n.item()) for n in l) + "\n"
            f.write(line)

    with open(observation_steps_fn, "w") as f:
        for i, l in enumerate(future.observation_steps):
            line = "\t".join(str(n.item()) for n in l) + "\n"
            f.write(line)

    with open(predicted_nodes_fn, "w") as f:
        for i, l in enumerate(future.predicted_nodes):
            line = "\t".join(str(n.item()) for n in l) + "\n"
            f.write(line)

    with open(predicted_times_fn, "w") as f:
        for i, l in enumerate(future.predicted_times):
            line = "\t".join(str(n.item()) for n in l) + "\n"
            f.write(line)

    with open(condition_fn, "w") as f:
        for i, l in enumerate(future.condition):
            f.write("{}\n".format(l))

    with open(trajectory_fn, "w") as f:
        for i in range(future.size):
            size = len(future.predicted_nodes[i])
            condition = future.condition[i]
            for j in range(size):
                node = future.predicted_nodes[i][j].item()
                time = future.predicted_times[i][j].item()
                node_id = graph.node_rid_map[node]
                time = time * config.obs_time_intervals
                f.write("{}\t{}\t{}\t{}\n".format(i, node_id, time, condition))
    return None


def predict(config, start_time, step_time, future, graph, trajectories, model):
    given_as_target, siblings_nodes = None, None

    def create_evaluator():
        return Evaluator(
            graph.n_node,
            given_as_target=given_as_target,
            siblings_nodes=siblings_nodes,
            config=config,
        )

    # モデルを使った予測
    for i in tqdm(range(0, len(trajectories), 1)):
        n_time_steps = int((start_time + step_time) / config.obs_time_intervals)
        evaluate(model, graph, trajectories, i, create_evaluator, future, n_time_steps)

    return None


def main_loop():
    start_time = 0
    step_time = 3600
    end_time = step_time * 6

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--name")
    args = parser.parse_args()

    # load configuration
    config = Config()
    config.load_from_file(args.config_file)

    chkpt_dir = os.path.join(config.workspace, config.checkpoint_directory, config.name)

    # load data
    graph, trajectories, model = pre_process(config)

    # loop
    future = Future(len(trajectories))
    for i in range(start_time, end_time, step_time):
        predict(config, i, step_time, future, graph, trajectories, model)
        write_prediction(config, future, chkpt_dir, graph)

    return None


if __name__ == "__main__":
    main_loop()
    print("end")
