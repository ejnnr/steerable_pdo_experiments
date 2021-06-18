import logging
import sys
import itertools
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

import matplotlib.pyplot as plt
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from e2cnn import nn

from diffop_experiments import MNISTRotModule


@hydra.main(config_path="config", config_name="config")
def cli_main(cfg: DictConfig):
    # Fix to prevent everything from being logged twice,
    # once by PL and once by Hydra.
    # See https://github.com/facebookresearch/hydra/issues/1012#issuecomment-806596005
    # This means that PL won't print its logs to console
    # but will hand them to Hydra, which then deals with logging.
    # We could instead only set pl_logger.propagate to False (without emptying
    # the handlers), but we want Hydra to log the output to files and in general
    # to configure the logging format.
    pl_logger = logging.getLogger("lightning")
    pl_logger.handlers = []
    pl_logger.propagate = True

    # allow addition of new keys
    OmegaConf.set_struct(cfg, False)

    if cfg.get("debug", False):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.weights_summary = "full"
        # speed up the debug run by using a tiny batch size
        cfg.data.batch_size = 2

    if cfg.get("full_debug", False):
        cfg.trainer.fast_dev_run = False
        cfg.trainer.max_steps = 1
        cfg.trainer.limit_val_batches = 2
        cfg.trainer.limit_test_batches = 2
        cfg.trainer.weights_summary = "full"
        cfg.data.batch_size = 2

    pl.seed_everything(cfg.seed)

    cfg.data.dir = hydra.utils.to_absolute_path(cfg.data.dir)

    # ------------
    # setup
    # ------------
    datamodule = hydra.utils.instantiate(cfg.data)

    if cfg.get("load_checkpoint", False):
        # If the load_checkpoint flag is passed, we load from that checkpoint.
        p = cfg.dir.log / Path(cfg.load_checkpoint)
        p = hydra.utils.get_original_cwd() / p
        # We don't use pytorch lightnings in-built LightningModule.load_from_checkpoint(),
        # instead we instantiate the model manually and load the state dict.
        # Using load_from_checkpoint() would require some ugly hacks to get the model type
        # (because we can't rely on hydra.utils.instantiate), though I'm not sure which
        # way is better
        if not torch.cuda.is_available():
            checkpoint = torch.load(p, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(p)

    cfg.model.input_size = datamodule.dims[1]
    cfg.model.in_channels = datamodule.dims[0]
    cfg.model.steps_per_epoch = datamodule.num_batches
    if cfg.trainer.stochastic_weight_avg:
        cfg.model.num_epochs = int(cfg.trainer.max_epochs * 0.8)
    else:
        cfg.model.num_epochs = cfg.trainer.max_epochs
    
    if cfg.get("load_checkpoint", False):
        # if we load weights anyway, no need to waste time on initialization
        cfg.model.init = None

    model = hydra.utils.instantiate(cfg.model)

    if cfg.get("load_checkpoint", False):
        # Now after instantiating the model, we actually load the state dict
        state_dict = checkpoint["state_dict"] # type: ignore
        model.load_state_dict(state_dict)
    
    if cfg.get("debug", False) or cfg.get("full_debug", False):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            print(name, p.numel())
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total number of trainable parameters: {num_params}")

    if cfg.get("eval_only", False):
        trainer = pl.Trainer(**cfg.trainer)
        results = trainer.test(model, datamodule=datamodule)
        return

    # ------------
    # training
    # ------------
    callbacks = []

    if cfg.get("log", True):
        # We want to always put tensorboard logs into the CWD,
        # no matter what cfg.dir.output_base is. The reason is that
        # on clusters, we use the scratch disk to save checkpoints,
        # but we want to make it easy to see the tensorboard logs
        # while the job is still running.
        tb_path = hydra.utils.to_absolute_path(cfg.dir.log + "/" + cfg.dir.run)
        # name and version should be empty; the path above is already a unique
        # path for this specific run, handled by Hydra
        logger = TensorBoardLogger(tb_path, name="", version="")
        callbacks.append(LearningRateMonitor())
    else:
        logger = None

    if cfg.data.validation_size:
        # checkpointing only makes sense if we use a validation set
        # (a final checkpoint for the last model is stored anyway)
        checkpoint_callback = ModelCheckpoint(
            monitor="loss/val",
            # the CWD is automatically set by Hydra, this is where
            # we want to save checkpoints
            dirpath=".",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    # we never want early stopping when we don't use a validations set
    if cfg.early_stopping.enabled and cfg.data.validation_size:
        early_stopping_callback = EarlyStopping(monitor="loss/val", patience=cfg.early_stopping.patience)
        callbacks.append(early_stopping_callback)
    
    # The logger directory might not be the CWD (see above), but we still
    # want to save weights there. This is only necessary for the case
    # where no validation set is used and thus no model checkpoint callback
    # (otherwise, the callback sets the correct path anyway)
    cfg.trainer.weights_save_path = "."
    # this doesn't play a large role, but I think it's used by the LR finder
    # even when the weights_save_path is set
    cfg.trainer.default_root_dir = "."

    if cfg.model.learning_rate == "auto" or cfg.get("only_find_lr", False):
        trainer = pl.Trainer(**cfg.trainer)
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        if cfg.get("only_find_lr", False):
            # in the only_find_lr setting, no tensorboard log is created, instead we store the figure
            fig.savefig("lr_plot.pdf")
        else:
            logger.experiment.add_figure("lr_finder", fig)
        model.hparams.learning_rate = lr_finder.suggestion()
        print("Best learning rate:", lr_finder.suggestion())
    
    if cfg.get("only_find_lr", False):
        return

    # we recreate the Trainer from scratch after determining the learning
    # rate. The reason is that Pytorch Lightning doesn't reset the epoch and step
    # count after tuning the learning rate. Could probably do this by hand,
    # but this seems more fool-proof.
    # This also avoids this issue: 
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5587
    # which is still unresolved at the time of writing this
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)

    # ------------
    # testing
    # ------------
    if (cfg.trainer.get("fast_dev_run", False)
        or not cfg.data.validation_size
        or cfg.trainer.stochastic_weight_avg):
        # In a fast dev run, no checkpoints will be created, we need to use the existing model.
        # If we don't use a validation set, we also can't load the best model
        # and need to use the last one.
        # And when using SWA, we want the averaged model, not one from a checkpoint.
        # (in the future, this might not be necessary: https://github.com/PyTorchLightning/pytorch-lightning/issues/6074)
        results = trainer.test(model, datamodule=datamodule)
    else:
        # otherwise, we load the best model.
        results = trainer.test(datamodule=datamodule)
    
    # write the test results into a file in the CWD
    # (which is handled by Hydra and is the same dir where the other
    # logs are stored)
    with open("results.yaml", "w") as file:
        # results is a list with a dict for each dataloader,
        # but we only use one test dataloader, so only print results[0]
        # default_flow_style just affects the style of YAML output
        yaml.dump(results[0], file, default_flow_style=False)


if __name__ == '__main__':
    cli_main()
