# pylint: disable=[E1101,W0621]

import os
import copy
import json
import warnings
from typing import List

import omegaconf
from omegaconf import OmegaConf

from conerf.evaluators.ace_zero_evaluator import AceZeroEvaluator
from conerf.utils.utils import setup_seed

warnings.filterwarnings("ignore", category=UserWarning)


def create_evaluator(
    config: OmegaConf,
    load_train_data: bool = False,
    trainset=None,
    load_val_data: bool = True,
    valset=None,
    load_test_data: bool = False,
    testset = None,
    models: List = None,
    meta_data: List = None,
    verbose: bool = False,
    device: str = "cuda",
):
    """Factory function for training neural network trainers."""
    if config.task == "pose":
        evaluator = AceZeroEvaluator(
            config, load_train_data, trainset,
            load_val_data, valset, load_test_data,
            testset, models, meta_data, verbose, device
        )
    else:
        raise NotImplementedError

    return evaluator


if __name__ == "__main__":
    from conerf.utils.config import config_parser, load_config
    args = config_parser()

    # parse YAML config to OmegaConf
    config = load_config(args.config)

    assert config.dataset.get("data_split_json", "") != "" or config.dataset.scene != ""

    setup_seed(config.seed)

    scenes = []
    if config.dataset.get("data_split_json", "") != "" and config.dataset.scene == "":
        # For objaverse only.
        with open(config.dataset.data_split_json, "r", encoding="utf-8") as fp:
            obj_id_to_name = json.load(fp)

        for idx, name in obj_id_to_name.items():
            scenes.append(name)
    elif (
        type(config.dataset.scene) == omegaconf.listconfig.ListConfig # pylint: disable=C0123
    ):  # pylint: disable=C0123
        scene_list = list(config.dataset.scene)
        for sc in config.dataset.scene:
            scenes.append(sc)
    else:
        scenes.append(config.dataset.scene)

    for scene in scenes:
        data_dir = os.path.join(config.dataset.root_dir, scene)
        if not os.path.exists(data_dir):
            continue

        local_config = copy.deepcopy(config)
        local_config.expname = (
            f"{config.neural_field_type}_{config.task}_{config.dataset.name}_{scene}"
        )
        local_config.expname = local_config.expname + "_" + args.suffix
        local_config.dataset.scene = scene

        evaluator = create_evaluator(
            local_config,
            load_train_data=False,
            trainset=None,
            load_val_data=True,
            valset=None,
            load_test_data=True,
            testset=None,
            verbose=True,
        )
        evaluator.eval(split="val")
        # evaluator.eval(split="test")
        evaluator.export_mesh()
