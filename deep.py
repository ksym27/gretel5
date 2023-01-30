import torch

def sampling_mask(observed, starts, targets, n_samples):
    n_observed = len(observed)
    n_indices = min(n_observed, n_samples)
    indices = torch.randperm(n_observed)[0:n_indices].sort().values
    observed = observed[indices, :]
    starts = starts[indices]
    targets = targets[indices]
    return observed, starts, targets
