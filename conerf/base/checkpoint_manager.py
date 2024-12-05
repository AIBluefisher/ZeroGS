import os
import shutil
import time

import torch


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class CheckPointManager:
    """
    Manager for saving/managing pytorch checkpoints.

    Provides functionality similar to tf.Saver such as
    max_to_keep and keep_checkpoint_every_n_hours
    """

    def __init__(
        self,
        save_path: str = None,
        max_to_keep: int = 5,
        keep_checkpoint_every_n_hours: float = 10000.0,
        verbose: bool = True
    ) -> None:
        if max_to_keep <= 0:
            raise ValueError('max_to_keep must be at least 1')

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self._verbose = verbose

        self._checkpoints_permanent = []  # Will not be deleted
        self._checkpoints_buffer = []  # Those which might still be deleted
        self._next_save_time = time.time()
        self._best_score = 0.0
        self._best_step = None

        if save_path is not None:
            self._save_path = save_path
            self._checkpoints_fname = os.path.join(
                save_path, 'checkpoints.txt')
            self._update_checkpoints_file()
        else:
            self._save_path = None
            self._checkpoints_fname = None

    def set_save_path(self, path: str):
        self._save_path = path

    def _save_checkpoint(
        self,
        step,
        models,
        optimizers: dict,
        schedulers: dict = None,
        meta_data: dict = None,
        score: float = 0.0,
    ):
        checkpoint_dir = os.path.join(self._save_path, 'model')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = os.path.join(
            checkpoint_dir, f'model_{step:06d}' + '.pth')

        state_dicts = {'step': step}

        # Append models.
        for model_name in models.keys():
            state_dicts[model_name] = de_parallel(
                models[model_name]).state_dict()

        # Append optimizers.
        for optim_name in optimizers.keys():
            state_dicts[optim_name] = optimizers[optim_name].state_dict()

        # Append schedulers.
        if schedulers is not None:
            for scheduler_name, scheduler in schedulers.items():
                if scheduler is not None:
                    state_dicts[scheduler_name] = scheduler.state_dict()

        # Append meta data.
        if meta_data is not None:
            for key in meta_data.keys():
                state_dicts[key] = meta_data[key]

        if self._verbose:
            print(f'Saving checkpoint: {checkpoint_name}')
        torch.save(state_dicts, checkpoint_name)

        # Track the latest model.
        shutil.copy(checkpoint_name, os.path.join(
            self._save_path, 'model.pth'))

        self._checkpoints_buffer.append((checkpoint_name, time.time()))

        # if np.all(np.array(score) >= np.array(self._best_score)):
        #     best_checkpoint_name = os.path.join(
        #         self._save_path, 'model_best.pth')
        #     shutil.copyfile(checkpoint_name, best_checkpoint_name)
        #     self._best_score = score
        #     self._best_step = step

        #     if self._verbose:
        #         print('Checkpoint is current best, score={}'.format(
        #             np.array_str(np.array(self._best_score), precision=3)))

    def _remove_old_checkpoints(self):
        while len(self._checkpoints_buffer) > self._max_to_keep:
            to_remove = self._checkpoints_buffer.pop(0)

            if to_remove[1] > self._next_save_time:
                self._checkpoints_permanent.append(to_remove)
                self._next_save_time = to_remove[1] + \
                    self._keep_checkpoint_every_n_hours * 3600
            else:
                os.remove(to_remove[0])

    def _update_checkpoints_file(self):
        checkpoints = [os.path.basename(c[0])
                       for c in self._checkpoints_permanent + self._checkpoints_buffer
                       ]
        with open(self._checkpoints_fname, 'w', encoding='utf-8') as fid:
            fid.write('\n'.join(checkpoints))
            fid.write(f'\nBest step: {self._best_step}')

    def save(
        self,
        models: dict,
        optimizers: dict,
        step: int,
        schedulers: dict = None,
        meta_data: dict = None,
        score: float = 0.0
    ):
        """
        Save model checkpoint to file

        Args:
            model: Torch model
            optimizer: Torch optimizer
            step (int): Step, model will be saved as model-[step].pth
            score (float, optional): To determine which model is the best
        """
        if self._save_path is None:
            raise AssertionError(
                'Checkpoint manager must be initialized with save path for save().')

        self._save_checkpoint(step, models, optimizers,
                              schedulers, meta_data, score)
        self._remove_old_checkpoints()
        self._update_checkpoints_file()

    def load_no_config(
        self,
        ckpt_path: str,
        distributed: bool = False,
        local_rank: int = 0,
        models: dict = None,
        optimizers: dict = None,
        schedulers: dict = None,
        meta_data: dict = None
    ) -> int:
        """
        Loads saved model from file

        Args:
            config: configurations
            models: Torch models to restore weights to
            optimizers: Optimizers
            schedulers: Schedulers
        """
        checkpoint_name = self._save_path
        if os.path.exists(ckpt_path):
            # Loaded from the specified checkpoint.
            if self._verbose:
                print(f'[INFO] Resuming from checkpoint {ckpt_path}...')
            checkpoint_name = ckpt_path
        elif os.path.isdir(self._save_path):
            # Loaded from the latest checkpoint.
            checkpoint_name = os.path.join(self._save_path, 'model.pth')
            if self._verbose:
                print(
                    f'[INFO] Resuming from latest checkpoint {checkpoint_name}...')
        if not os.path.exists(checkpoint_name):
            if self._verbose:
                print(f"[WARNING] Checkpoint {checkpoint_name} does not exist, " + \
                      "training from scratch!")
            return 0

        if distributed:
            state = torch.load(
                checkpoint_name, map_location=f'cuda:{local_rank}')
        else:
            state = torch.load(checkpoint_name, map_location='cpu')

        step = 0
        if 'step' in state:
            step = state['step']

        if models is not None:
            for model_name in models.keys():
                models[model_name].load_state_dict(state[model_name])

        if optimizers is not None:
            for optim_name in optimizers.keys():
                optimizers[optim_name].load_state_dict(state[optim_name])

        if schedulers is not None:
            for scheduler_name, scheduler in schedulers.items():
                if scheduler is not None:
                    scheduler.load_state_dict(state[scheduler_name])

        if meta_data is not None:
            for key in meta_data.keys():
                meta_data[key] = state[key]

        if self._verbose:
            print(f'[INFO] Loaded models from {checkpoint_name}')

        return step

    def load(
        self,
        config,
        models: dict = None,
        optimizers: dict = None,
        schedulers: dict = None,
        meta_data: dict = None
    ) -> int:
        """
        Loads saved model from file

        Args:
            config: configurations
            models: Torch models to restore weights to
            optimizers: Optimizers
            schedulers: Schedulers
        """
        return self.load_no_config(
            config.ckpt_path,
            config.distributed,
            config.local_rank,
            models,
            optimizers,
            schedulers,
            meta_data
        )
