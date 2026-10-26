"""Solution errors computed in float64 on the evaluation set."""
import torch


def solution_errors(prediction, reference):
    """Return pooled RMSE, MAE, max error and relative L2 over all entries.

    Scalar fields may use (N,) or (N, 1). No other broadcasting is allowed.
    Relative L2 is None when the reference norm is zero (undefined).
    """
    prediction = torch.as_tensor(prediction).detach().cpu().double()
    reference = torch.as_tensor(reference).detach().cpu().double()
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if reference.ndim == 1:
        reference = reference[:, None]
    if prediction.shape != reference.shape or prediction.ndim != 2 or not prediction.numel():
        raise ValueError("Prediction and reference must have matching non-empty (N, C) shapes.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("Prediction and reference must be finite.")
    error = prediction - reference
    norm = torch.linalg.vector_norm(reference).item()
    return {
        "rmse": error.square().mean().sqrt().item(),
        "mae": error.abs().mean().item(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2": torch.linalg.vector_norm(error).item() / norm if norm else None,
    }
