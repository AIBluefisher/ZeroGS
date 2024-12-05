import torch


class ModelBase(torch.nn.Module):
    """
    An abstract class which defines some basic operations for a torch model.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def to_distributed(self):
        """Change model to distributed mode."""
        raise NotImplementedError

    def switch_to_eval(self):
        """Change model to evaluation mode."""
        raise NotImplementedError

    def switch_to_train(self):
        """Change model to training mode."""
        raise NotImplementedError

    def forward(self, data, **kwargs):
        raise NotImplementedError
