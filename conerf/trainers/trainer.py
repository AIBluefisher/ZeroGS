import os
import socket

import torch
import tqdm
import visdom

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from conerf.base.checkpoint_manager import CheckPointManager


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()

    return is_open


class BaseTrainer:
    def __init__(
        self, config: OmegaConf, prefetch_dataset: bool = True, trainset=None, valset=None
    ) -> None:
        super().__init__()

        self.trainer_name = 'BaseTrainer'
        self.config = config
        self.device = "cuda" # f"cuda:{config.trainer.local_rank}"

        self.output_path = os.path.join(
            config.dataset.root_dir, 'out', config.expname
        )
        if self.config.trainer.local_rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
            print(f'[INFO] Outputs will be saved to {self.output_path}')

        self.scheduler = None
        self.model = None
        self.scalars_to_log = dict()
        self.images_to_log = dict()
        self.ckpt_manager = CheckPointManager(
            save_path=self.output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5,
            verbose=config.get('verbose', False),
        )

        self.train_done = False
        self.state_dicts = None
        self.epoch = 0
        self.iteration = 0

        if prefetch_dataset:
            self.load_dataset()
        else:
            self.train_dataset = trainset
            self.val_dataset = valset

        self._setup_visualizer()
        self.setup_training_params()

        # Functions need to be overwritten.
        self.build_networks()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.compose_state_dicts()

    def __del__(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _check(self):
        assert self.train_dataset is not None
        assert self.val_dataset is not None
        assert self.model is not None
        assert os.path.exists(self.output_path) is True

        if self.config.trainer.enable_tensorboard and self.config.trainer.local_rank == 0:
            assert self.writer is not None

        if self.config.trainer.enable_visdom and self.config.trainer.local_rank == 0:
            assert self.visdom is not None

    def load_dataset(self):
        raise NotImplementedError

    def _setup_visualizer(self):
        self.writer = None
        self.visdom = None

        # Setup tensorboard.
        if self.config.trainer.enable_tensorboard and self.config.trainer.local_rank == 0:
            log_dir = os.path.join(
                self.config.dataset.root_dir, 'logs', self.config.expname)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # Setup visdom.
        if self.config.trainer.enable_visdom and self.config.trainer.local_rank == 0:
            # check if visdom server is running
            is_open = check_socket_open(
                self.config.trainer.visdom_server, self.config.trainer.visdom_port)
            retry = None
            while not is_open:
                retry = input(
                    f"visdom port ({self.config.trainer.visdom_port}) not open, retry? (y/n) ")
                if retry not in ["y", "n"]:
                    continue
                if retry == "y":
                    is_open = check_socket_open(
                        self.config.trainer.visdom_server, self.config.trainer.visdom_port)
                else:
                    break

            self.visdom = visdom.Visdom(
                server=self.config.trainer.visdom_server,
                port=self.config.trainer.visdom_port,
                env='conerf'
            )
            print('[INFO] Visualizing camera poses at ' + \
                  f'{self.config.trainer.visdom_server}:{self.config.trainer.visdom_port}')

    def setup_training_params(self):
        pass
    
    def build_networks(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def setup_optimizer(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def setup_loss_functions(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def train(self):
        pbar = tqdm.trange(self.config.n_iters,
                           desc=f"Training {self.config.expname}", leave=False)

        iter_start = self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                                          load_scheduler=not self.config.no_load_scheduler)

        if self.config.distributed:
            # NOTE: Distributed mode can only be activated after loading models.
            self.model.to_distributed()

        while self.iteration < iter_start:
            pbar.update(1)
            self.iteration += 1

        while self.iteration < self.config.n_iters + 1:
            for self.train_data in self.train_loader: # pylint: disable=[E1101,W0201]
                # Main training logic.
                self.train_iteration(data_batch=self.train_data)

                if self.config.local_rank == 0:
                    # Main validation logic.
                    if self.iteration % self.config.n_validation == 0:
                        score = self.validate()

                    # log to tensorboard.
                    if self.iteration % self.config.n_tensorboard == 0:
                        self.log_info()

                    # save checkpoint.
                    if self.iteration % self.config.n_checkpoint == 0:
                        score = self.validate()
                        self.save_checkpoint(score=score)

                pbar.update(1)

                self.iteration += 1
                if self.iteration > self.config.n_iters + 1:
                    break
            self.epoch += 1

        self.train_done = True

    def train_iteration(self, data_batch) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> float: # pylint: disable=R0201
        score = 0.
        # self.model.switch_to_eval()
        # ... (implement validation logic here)
        # self.model.switch_to_train()

        return score

    def compose_state_dicts(self) -> None:
        """
            Implement this function and follow the format below:
            self.state_dicts = {'models': None, 'optimizers': None, 'schedulers': None}
        """

        raise NotImplementedError

    @torch.no_grad()
    def log_info(self) -> None:
        if self.writer is None:
            return

        log_str = f'{self.config.expname} Epoch: {self.epoch}  step: {self.iteration} '

        for key in self.scalars_to_log:
            log_str += ' {}: {:.6f}'.format(key, self.scalars_to_log[key])
            self.writer.add_scalar(
                key, self.scalars_to_log[key], self.iteration)

        # print(log_str, file=self.log_file)
    
    @torch.no_grad()
    def log_images(self) -> None:
        for key in self.images_to_log:
            if self.images_to_log[key].dim() == 3:
                self.writer.add_image(key, self.images_to_log[key], self.iteration)
            else:
                self.writer.add_images(key, self.images_to_log[key], self.iteration)

    def save_checkpoint(self, score: float = 0.0) -> None:
        assert self.state_dicts is not None

        self.ckpt_manager.save(
            models=self.state_dicts['models'],
            optimizers=self.state_dicts['optimizers'],
            schedulers=self.state_dicts['schedulers'],
            meta_data=self.state_dicts['meta_data'],
            step=self.iteration,
            score=score
        )

    def load_checkpoint(
        self,
        load_model=True,
        load_optimizer=True,
        load_scheduler=True,
        load_meta_data=False
    ) -> int:
        iter_start = self.ckpt_manager.load(
            config=self.config.trainer,
            models=self.state_dicts['models'] if load_model else None,
            optimizers=self.state_dicts['optimizers'] if load_optimizer else None,
            schedulers=self.state_dicts['schedulers'] if load_scheduler else None,
            meta_data=self.state_dicts['meta_data'] if load_meta_data else None
        )

        return iter_start
