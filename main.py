import argparse
import os
import sys
import time

from torch import Tensor

import numpy as np
import torch
from termcolor import colored
from typing import Optional, List, Callable, Tuple, Union, Any

from config import Config
from graph import Graph
from metrics import Evaluator
from model import Model, MLP, MultiDiffusion, EdgeTransformer
from trajectories import Trajectories
from utils import generate_masks, sampling_mask

""" ======= DATA LOADING ======= """


def load_tensor(device: torch.device, path: str, *subpaths) -> Optional[torch.Tensor]:
    tensor = None
    filename = os.path.join(path, *subpaths)
    if os.path.exists(filename):
        tensor = torch.load(filename)
        tensor = tensor.to(device)
    return tensor


def load_data2(
        config: Config,
        input_dir2: str
) -> Tuple[Union[Graph, Any], Tuple[Any, Any, Any], Optional[Tensor], Optional[Tensor]]:
    """Read data in config.workspace / config.input_directory

    Args:
        config (Config): configuration
        input_dir (str): input directory name
    Returns:
        (Graph, List[Trajectories], torch.Tensor, torch.Tensor):
            graph, (train, valid, test)_trajectories, pairwise_node_features, pairwise_distances
    """

    input_dir = os.path.join(config.workspace, config.input_directory)
    graph = Graph.read_from_files_for_deep(
        nodes_filename=os.path.join(input_dir, "nodes.txt"),
        edges_filename=os.path.join(input_dir, "edges.txt")
    )

    # データの読み込み先の設定
    lengths_filename = os.path.join(input_dir2, "lengths.txt")
    observations_filename = os.path.join(input_dir2, "observation_6sec.txt")
    paths_filename = os.path.join(input_dir2, "paths.txt")

    #
    trajectories = Trajectories.read_from_files_for_deep(
        lengths_filename=lengths_filename,
        observations_filename=observations_filename,
        num_nodes=graph.n_node,
        node_id_map=graph.node_id_map,
        graph=graph,
        obs_time_intervals=config.obs_time_intervals,
        paths_filename = paths_filename
    )

    trajectories = trajectories.to(config.device)

    return graph, trajectories


def load_data(
        config: Config,
) -> Tuple[Union[Graph, Any], Tuple[Any, Any, Any], Optional[Tensor], Optional[Tensor]]:
    """Read data in config.workspace / config.input_directory

    Args:
        config (Config): configuration

    Returns:
        (Graph, List[Trajectories], torch.Tensor, torch.Tensor):
            graph, (train, valid, test)_trajectories, pairwise_node_features, pairwise_distances
    """

    input_dir = os.path.join(config.workspace, config.input_directory)

    graph = Graph.read_from_files_for_deep(
        nodes_filename=os.path.join(input_dir, "nodes.txt"),
        edges_filename=os.path.join(input_dir, "edges.txt"),
        blockage_filename=os.path.join(input_dir, "blockage.txt"),
        blockage_time_intervals=config.blockage_time_intervals
    )

    # データの読み込み先の設定
    lengths_filename = os.path.join(input_dir, "lengths_s.txt")
    observations_filename = os.path.join(input_dir, "observation_6sec_s.txt")
    paths_filename = os.path.join(input_dir, "paths_s.txt")

    #
    trajectories = Trajectories.read_from_files_for_deep(
        lengths_filename=lengths_filename,
        observations_filename=observations_filename,
        num_nodes=graph.n_node,
        node_id_map=graph.node_id_map,
        graph=graph,
        obs_time_intervals=config.obs_time_intervals,
        paths_filename=paths_filename
    )

    pairwise_node_features = load_tensor(config.device, input_dir, "pairwise_node_features.pt")
    pairwise_distances = load_tensor(config.device, input_dir, "shortest-path-distance-matrix.pt")

    trajectories.pairwise_node_distances = pairwise_distances

    if config.extract_coord_features:
        print("Node coordinates are removed from node features")
        graph.extract_coords_from_features(keep_in_features=False)

    valid_trajectories_mask = trajectories.lengths >= config.min_trajectory_length
    valid_trajectories_mask &= trajectories.lengths <= config.max_trajectory_length
    valid_trajectories_idx = valid_trajectories_mask.nonzero()[:, 0]
    valid_lengths = trajectories.lengths[valid_trajectories_idx]

    print("number of trajectories: ", len(trajectories))
    print(
        f"number of valid trajectories (length in [{config.min_trajectory_length}, {config.max_trajectory_length}]): {len(valid_trajectories_idx)}"
    )
    print(
        f"trajectories length: min {valid_lengths.min()} | max {valid_lengths.max()} | mean {valid_lengths.float().mean():.2f}"
    )

    trajectories = trajectories.to(config.device)

    if config.overfit1:
        config.batch_size = 1
        id_ = (trajectories.lengths == config.number_observations + 1).nonzero()[0]
        print(f"Overfit on trajectory {id_.item()} of length {trajectories.lengths[id_].item()}")
        train_mask = torch.zeros_like(valid_trajectories_mask)
        train_mask[id_] = 1
        test_mask = valid_mask = train_mask

    else:
        print(f"split train/(valid)?/test {config.train_test_ratio}")
        proportions = list(map(float, config.train_test_ratio.split("/")))
        if len(proportions) == 2:
            train_prop, test_prop = proportions
            valid_prop = 0.0
        elif len(proportions) == 3:
            train_prop, valid_prop, test_prop = proportions

        n_train = int(train_prop * len(valid_trajectories_idx))
        n_valid = int(valid_prop * len(valid_trajectories_idx))
        n_test = int(test_prop * len(valid_trajectories_idx))

        # 念の為、シャッフルしておく。
        torch.random.manual_seed(20)
        idx = torch.randperm(valid_trajectories_idx.shape[0])
        valid_trajectories_idx = valid_trajectories_idx[idx].view(-1)

        train_idx = valid_trajectories_idx[:n_train]
        train_mask = torch.zeros_like(valid_trajectories_mask)
        train_mask[train_idx] = 1

        valid_idx = valid_trajectories_idx[n_train: n_train + n_valid]
        valid_mask = torch.zeros_like(valid_trajectories_mask)
        valid_mask[valid_idx] = 1

        test_idx = valid_trajectories_idx[n_train + n_valid: n_train + n_valid + n_test]
        test_mask = torch.zeros_like(valid_trajectories_mask)
        test_mask[test_idx] = 1

    train_trajectories = trajectories.with_mask(train_mask)
    valid_trajectories = trajectories.with_mask(valid_mask)
    test_trajectories = trajectories.with_mask(test_mask)
    trajectories = (train_trajectories, valid_trajectories, test_trajectories)

    return graph, trajectories, pairwise_node_features, pairwise_distances


def display_baseline(
        config: Config,
        graph: Graph,
        train_trajectories: Trajectories,
        test_trajectories: Trajectories,
        evaluator: Evaluator,
):
    """Compute baseline uniform random walk with/without

    Args:
        config (Config): [description]
        graph (Graph): graph
        train_trajectories (Trajectories): train trajectories
        test_trajectories (Trajectories): test trajectories
        evaluator (Evaluator): evaluator to compute metrics
    """

    graph = graph.add_self_loops(degree_zero_only=True)
    graph = graph.update(edges=torch.ones(graph.n_edge, device=graph.device))
    graph = graph.softmax_weights()
    print("Computing non backtracking edges...")
    graph.compute_non_backtracking_edges()
    print("Done")

    print("=== BASELINE ===")
    baseline_model = create_baseline(config, non_backtracking=False)

    print("TEST DATASET")
    evaluator.compute(baseline_model, graph, test_trajectories, None)
    print(colored(evaluator.to_string(), "green"))

    print("TRAIN DATASET")
    evaluator.compute(baseline_model, graph, train_trajectories, None)
    print(colored(evaluator.to_string(), "green"))

    print("=== NON BACKTRACKING BASELINE ===")
    nb_baseline_model = create_baseline(config, non_backtracking=True)

    print("TEST DATASET")
    evaluator.compute(nb_baseline_model, graph, test_trajectories, None)
    print(colored(evaluator.to_string(), "green"))

    print("TRAIN DATASET")
    evaluator.compute(nb_baseline_model, graph, train_trajectories, None)
    print(colored(evaluator.to_string(), "green"))


def compute_loss(
        typ: str,
        trajectories: Trajectories,
        observations: torch.Tensor,
        predictions: torch.Tensor,
        starts: torch.Tensor,
        targets: torch.Tensor,
        rw_weights: torch.Tensor,
        trajectory_idx: int,
):
    """Compute the †raining lossread_from_files

    Args:
        typ (str): loss flag from configuration, can be RMSE, dot_loss, log_dot_loss, target_only or nll_loss
        trajectories (Trajectories): full trajectories dataset evaluated
        observations (torch.Tensor): current trajectory observation [traj_length, n_node]
        predictions (torch.Tensor): output prediction of the model [n_pred, n_node]
        starts (torch.Tensor): indexes of starts extrapolation in observations [n_pred,]
        targets (torch.Tensor):  [n_pred,]
        rw_weights (torch.Tensor): random walk weights output of model [n_pred, n_edge]
        trajectory_idx (int): index of evaluated trajectory

    Returns:
        torch.Tensor(): loss for this prediction
    """

    if typ == "RMSE":
        return ((predictions - observations[targets]) ** 2).sum()
    elif typ == "dot_loss":
        return -1.0 * (predictions * observations[targets]).sum()
    elif typ == "log_dot_loss":
        return -1.0 * ((predictions * observations[targets]).sum(dim=1) + 1e-30).log().sum()
    elif typ == "target_only":
        return -predictions[observations[targets] > 0].sum()
    elif typ == "nll_loss":
        loss = torch.tensor(0.0, device=trajectories.device)
        log_rw_weights = -(rw_weights + 1e-20).log()
        for pred_id in range(len(starts)):
            for jump_id in range(starts[pred_id], targets[pred_id]):
                traversed_edges = trajectories.traversed_edges(trajectory_idx, jump_id)
                loss += log_rw_weights[pred_id, traversed_edges].sum()
        return loss
    else:
        raise Exception(f'Unknown loss "{typ}"')


def create_optimizer(params, config: Config) -> torch.optim.Optimizer:
    """Create the torch optimizer, config.optimizer can be SGD, Adam or RMSprop"""
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=config.lr)
    elif config.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=config.lr)
    else:
        raise Exception(f"Unknown optimizer '{config.optimizer}''")
    return optimizer


def create_baseline(config: Config, non_backtracking: bool):
    return Model(
        diffusion_graph_transformer=None,
        multichannel_diffusion=None,
        direction_edge_mlp=None,
        number_observations=config.number_observations,
        rw_expected_steps=config.rw_expected_steps,
        rw_non_backtracking=non_backtracking,
        latent_transformer_see_target=False,
        double_way_diffusion=False,
        diffusion_self_loops=False,
    )


def create_model(graph: Graph, cross_features: Optional[torch.Tensor], config: Config) -> Model:
    """Create an instance of Gretel

    Args:
        graph (Graph): graph
        cross_features ([type]): available cross features between nodes [n_node, n_node, d_cross]
            Can be useful to show distance with target.
        config (Config): configuration

    Returns:
        Model: the Gretel
    """

    def dimension(tensor, name):
        if tensor is None:
            return 0
        if tensor.dim() == 1:
            return 1
        elif tensor.dim() == 2:
            return tensor.shape[1]
        else:
            raise ValueError(f"{name} features should be scalar or vectors")

    d_node = dimension(graph.nodes, "graph.nodes")
    d_edge = dimension(graph.edges, "graph.edges") + 1  # 閉鎖属性を追加する
    d_cross = cross_features.shape[2] if cross_features is not None else 0

    diffusion_graph_transformer = None
    if config.initial_edge_transformer and (d_node > 0 or d_edge > 0):
        diffusion_graph_transformer = EdgeTransformer(d_node, d_edge, 1)
    else:
        print("No initial edge transformer.")

    multichannel_diffusion = MultiDiffusion(
        config.diffusion_k_hops, config.diffusion_hidden_dimension, config.parametrized_diffusion
    )

    double_way_diffusion = 2 if config.double_way_diffusion else 1
    d_in_direction_mlp = (
            2 * config.number_observations * config.diffusion_hidden_dimension * double_way_diffusion
            + 2 * d_node
            + d_edge
            + (d_node if config.latent_transformer_see_target else 0)
            + (2 * d_cross if config.latent_transformer_see_target else 0)
            + 1
    )
    direction_edge_mlp = MLP(d_in_direction_mlp, 1)

    return Model(
        diffusion_graph_transformer=diffusion_graph_transformer,
        multichannel_diffusion=multichannel_diffusion,
        direction_edge_mlp=direction_edge_mlp,
        number_observations=config.number_observations,
        rw_expected_steps=config.rw_expected_steps,
        rw_non_backtracking=config.rw_non_backtracking,
        latent_transformer_see_target=config.latent_transformer_see_target,
        double_way_diffusion=config.double_way_diffusion,
        diffusion_self_loops=config.diffusion_self_loops,
    )


def train_epoch(
        model: Model,
        graph: Graph,
        optimizer: torch.optim.Optimizer,
        config: Config,
        train_trajectories: Trajectories,
        pairwise_node_features: torch.Tensor,
):
    """One epoch of training"""
    model.train()

    print_cum_loss = 0.0
    print_num_preds = 0
    print_time = time.time()
    print_every = len(train_trajectories) // config.batch_size // config.print_per_epoch

    trajectories_shuffle_indices = np.arange(len(train_trajectories))
    if config.shuffle_samples:
        np.random.shuffle(trajectories_shuffle_indices)

    for iteration, batch_start in enumerate(
            range(0, len(trajectories_shuffle_indices) - config.batch_size + 1, config.batch_size)
    ):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=config.device)

        for i in range(batch_start, batch_start + config.batch_size):
            trajectory_idx = trajectories_shuffle_indices[i]
            observations = train_trajectories[trajectory_idx]

            number_steps = None
            if config.rw_edge_weight_see_number_step or config.rw_expected_steps:
                if config.use_shortest_path_distance:
                    number_steps = (
                            train_trajectories.leg_shortest_lengths(trajectory_idx).float() * 1.1
                    ).long()
                else:
                    number_steps = train_trajectories.leg_lengths(trajectory_idx)

            observed, starts, targets = generate_masks(
                trajectory_length=observations.shape[0],
                number_observations=config.number_observations,
                predict=config.target_prediction,
                with_interpolation=config.with_interpolation,
                device=config.device,
            )

            # # 間引いたマスクを生成する
            # observed, starts, targets = sampling_mask(observed, starts, targets, config.num_observed_samples)
            # 時間情報を取得
            node_times = train_trajectories.times(trajectory_idx)

            # 観測時間の閉塞データを取得する
            blockage = graph.blockage[:, node_times]

            lp_graph = graph.update(edges=torch.cat([graph.edges, torch.unsqueeze(blockage[:, 0], dim=1)], dim=1))

            #
            diffusion_graph = lp_graph if not config.diffusion_self_loops else lp_graph.add_self_loops()

            predictions, potentials, rw_weights = model(
                observations,
                lp_graph,
                diffusion_graph,
                observed=observed,
                starts=starts,
                targets=targets,
                pairwise_node_features=pairwise_node_features,
                number_steps=number_steps,
                blockage=blockage
            )

            print_num_preds += starts.shape[0]

            l = (
                    compute_loss(
                        config.loss,
                        train_trajectories,
                        observations,
                        predictions,
                        starts,
                        targets,
                        rw_weights,
                        trajectory_idx,
                    )
                    / starts.shape[0]
            )
            loss += l

        loss /= config.batch_size
        print_cum_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % print_every == 0:
            print_loss = print_cum_loss / print_every
            print_loss /= print_num_preds
            pred_per_second = 1.0 * print_num_preds / (time.time() - print_time)

            print_cum_loss = 0.0
            print_num_preds = 0
            print_time = time.time()

            progress_percent = int(
                100.0 * ((iteration + 1) // print_every) / config.print_per_epoch
            )

            print(
                f"Progress {progress_percent}% | iter {iteration} | {pred_per_second:.1f} pred/s | loss {config.loss} {print_loss}"
            )


def evaluate(
        model,
        graph,
        trajectories,
        pairwise_node_features,
        evaluator_creator: Callable[[], Evaluator],
        dataset: str = "TRAIN",
) -> Evaluator:
    print(f"\n=== {dataset} ===\n")
    model.eval()
    evaluator = evaluator_creator()
    evaluator.compute(model, graph, trajectories, pairwise_node_features)
    print(colored(evaluator.to_string(), "red"))
    return evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--name")
    args = parser.parse_args()

    # load configuration
    config = Config()
    config.load_from_file(args.config_file)

    graph, trajectories, pairwise_node_features, _ = load_data(config)
    train_trajectories, valid_trajectories, test_trajectories = trajectories
    use_validation_set = len(valid_trajectories) > 0
    graph = graph.to(config.device)

    given_as_target, siblings_nodes = None, None
    # if config.dataset == "wikispeedia":
    #     given_as_target, siblings_nodes = load_wiki_data(config)

    if pairwise_node_features is not None:
        pairwise_node_features = pairwise_node_features.to(config.device)

    if config.rw_edge_weight_see_number_step:  # TODO
        raise NotImplementedError

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

    if config.restore_from_checkpoint:
        chkpt_file_path = os.path.join(config.workspace, config.checkpoint_directory, config.name,
                                       config.checkpoint_file_name)
        checkpoint_data = torch.load(chkpt_file_path)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        print("Loaded parameters from checkpoint")

    def create_evaluator():
        return Evaluator(
            graph.n_node,
            given_as_target=given_as_target,
            siblings_nodes=siblings_nodes,
            config=config,
        )

    if use_validation_set:
        valid_evaluator = Evaluator(
            graph.n_node,
            given_as_target=given_as_target,
            siblings_nodes=siblings_nodes,
            config=config,
        )

    if config.compute_baseline:
        display_baseline(config, graph, train_trajectories, test_trajectories, create_evaluator())

    graph = graph.add_self_loops(
        degree_zero_only=config.self_loop_deadend_only, edge_value=config.self_loop_weight
    )

    if config.rw_non_backtracking:
        print("Computing non backtracking graph...", end=" ")
        sys.stdout.flush()
        graph.compute_non_backtracking_edges()
        print("Done")

    evaluate(
        model, graph, test_trajectories, pairwise_node_features, create_evaluator, dataset="TEST"
    )

    if use_validation_set:
        evaluate(
            model,
            graph,
            valid_trajectories,
            pairwise_node_features,
            create_evaluator,
            dataset="EVAL",
        )

    for epoch in range(config.number_epoch):

        print(f"\n=== EPOCH {epoch} ===")

        model.train()
        train_epoch(model, graph, optimizer, config, train_trajectories, pairwise_node_features)

        # VALID and TEST metrics computation
        test_evaluator = evaluate(
            model,
            graph,
            test_trajectories,
            pairwise_node_features,
            create_evaluator,
            dataset="TEST",
        )
        #
        # valid_evaluator = None
        # if use_validation_set:
        #     valid_evaluator = evaluate(
        #         model,
        #         graph,
        #         valid_trajectories,
        #         pairwise_node_features,
        #         create_evaluator,
        #         dataset="EVAL",
        #     )

        if config.enable_checkpointing and epoch % config.checkpoint_every_num_epoch == 0:
            print("Checkpointing...")
            directory = os.path.join(config.workspace, config.checkpoint_directory, config.name)
            chkpt_file = os.path.join(directory, f"{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                chkpt_file,
            )
            config_file = os.path.join(directory, "config")
            config.save_to_file(config_file)

            metrics_file = os.path.join(directory, f"{epoch:04d}.txt")
            with open(metrics_file, "w") as f:
                f.write(test_evaluator.to_string())
            #     if valid_evaluator:
            #         f.write("\n\n=== VALIDATION ==\n\n")
            #         f.write(valid_evaluator.to_string())

            print(colored(f"Checkpoint saved in {chkpt_file}", "blue"))

    # print(test_evaluator.get_metrics())
    print("end")


if __name__ == "__main__":
    main()
