from omegaconf import OmegaConf

from conerf.base.model_base import ModelBase
from conerf.trainers.ace_zero_trainer import AceZeroTrainer


def create_trainer(
    config: OmegaConf,
    prefetch_dataset=True,
    trainset=None,
    valset=None,
    model: ModelBase = None
):
    """Factory function for training neural network trainers."""
    if config.task == "pose":
        trainer = AceZeroTrainer(config, prefetch_dataset, trainset, valset)
    else:
        raise NotImplementedError

    return trainer
