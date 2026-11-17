import torch

_LOSS_METRIC_FACTORIES = {
    "MSE": lambda: torch.nn.MSELoss(reduction="mean"),
}

loss_metric_dict = {
    "MSE": torch.nn.MSELoss(reduction="mean")
}

optimiser_dict = {
    "adam": torch.optim.Adam
}


def loss_metric(loss_metric_string):
    """
    Returns a fresh loss metric module.
    """
    if loss_metric_string in _LOSS_METRIC_FACTORIES:
        return _LOSS_METRIC_FACTORIES[loss_metric_string]()
    if loss_metric_string in loss_metric_dict:
        return loss_metric_dict[loss_metric_string]
    raise ValueError("Invalid loss metric.")
    
def choose_optimiser(optimiser_string):
    """
    Returns the optimiser module.
    """
    if optimiser_string in optimiser_dict:
        return optimiser_dict[optimiser_string]
    else:
        raise ValueError("Invalid optimiser.")