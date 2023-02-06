import torch


def sampling_mask(observed, starts, targets, n_samples):
    n_observed = len(observed)
    n_indices = min(n_observed, n_samples)
    indices = torch.randperm(n_observed)[0:n_indices].sort().values
    observed = observed[indices, :]
    starts = starts[indices]
    targets = targets[indices]
    return observed, starts, targets

#
# if torch.count_nonzero(prediction, 0) != 0:
#     # enabled_edge = torch.where(graph.senders == current_node)
#     # blockage_edge = torch.where(blockage[:,-1] == 0)
#     #
#     # enabled_edge = torch.tensor([e for e in enabled_edge[0] if e in blockage_edge[0]], device=config.device)
#     # enabled_node = graph.receivers[enabled_edge]
#     #
#     # prediction = prediction[enabled_node]
#     # index = prediction.multinomial(num_samples=1, replacement=True)
#     #
#     # next_node = enabled_node[index].squeeze()
#
#     next_node = prediction.multinomial(num_samples=1, replacement=True)
#     next_node = next_node[0]