import torch


def flat_accuracy(ground_truth: torch.Tensor, predictions: torch.Tensor):
    predictions_flat = torch.argmax(predictions, dim=-1).flatten()
    ground_truth_flat = torch.flatten(ground_truth)
    return torch.sum(predictions_flat == ground_truth_flat) / len(ground_truth_flat)


def dummy(ground_truth: torch.Tensor, predictions: torch.Tensor):
    return 42


METRICS = {
    'flat_accuracy': flat_accuracy,
    'dummy': dummy
}
