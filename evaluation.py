import argparse
import os
import sys

import torch

from typing import Callable
from config import Config
from metrics import Evaluator
from main import load_data, create_optimizer, create_model, evaluate


def evaluate(
    model,
    graph,
    trajectories,
    trajectory_idx,
    evaluator_creator: Callable[[], Evaluator]
) -> Evaluator:
    model.eval()
    evaluator = evaluator_creator()
    return evaluator.test_compute(model, graph, trajectories, trajectory_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--name")
    args = parser.parse_args()

    # load configuration
    config = Config()
    config.load_from_file(args.config_file)

    graph, trajectories, pairwise_node_features, _ = load_data(config)
    train_trajectories, _, test_trajectories = trajectories
    graph = graph.to(config.device)

    given_as_target, siblings_nodes = None, None

    if args.name is not None:
        print(f"Experiment name from CLI: {args.name}")
        config.name = args.name

    if not config.name:
        experiment_name = input("Give a name to this experiment? ").strip()
        config.name = experiment_name or config.date

    print(f'==== START "{config.name}" ====')

    torch.manual_seed(config.seed)

    if config.enable_checkpointing:
        chkpt_dir = os.path.join(config.workspace, config.checkpoint_directory, config.name)
        os.makedirs(chkpt_dir, exist_ok=True)
        print(f"Checkpoints will be saved in [{chkpt_dir}]")

    d_node = graph.nodes.shape[1] if graph.nodes is not None else 0
    d_edge = graph.edges.shape[1] if graph.edges is not None else 0
    print(f"Number of node features {d_node}. Number of edge features {d_edge}")

    model = create_model(graph, pairwise_node_features, config)
    model = model.to(config.device)

    optimizer = create_optimizer(model.parameters(), config)

    # ここはオプション化する必要がある。
    filename = 'C:/Users/kashiyama/PycharmProjects/gretel3/workspace/chkpt/planar-lr.1-interpolation-nb-nll/0030.pt'
    checkpoint_data = torch.load(filename)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    def create_evaluator():
        return Evaluator(
            graph.n_node,
            given_as_target=given_as_target,
            siblings_nodes=siblings_nodes,
            config=config,
        )

    graph = graph.add_self_loops(
        degree_zero_only=config.self_loop_deadend_only, edge_value=config.self_loop_weight
    )

    if config.rw_non_backtracking:
        print("Computing non backtracking graph...", end=" ")
        sys.stdout.flush()
        graph.compute_non_backtracking_edges()
        print("Done")

    # モデルを使った予測
    future = evaluate(
        model,
        graph,
        test_trajectories,
        0,
        create_evaluator
    )
    # 予測のPrefix
    for i, x in enumerate(future.observations):
        print(int(torch.nonzero(x)[0][0]),',')

    # 予測のSurffix
    with open('./result.csv', 'w') as f:
        for i, x in enumerate(future.prediction):
            f.write("{},{}\n".format(i, x))

    print("end")

if __name__ == "__main__":
    main()
