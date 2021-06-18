from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from e2cnn import gspaces, nn

from .utils import build_optimizer_sfcnn

class Network(pl.LightningModule, ABC):
    def __init__(self,
                 n_classes: int = 10,
                 input_size: int = 28,
                 learning_rate: float = 1e-3,
                 in_channels: int = 1,
                 conv_dropout: float = 0.0,
                 fc_dropout: float = 0.0,
                 optimizer: str = "adam",
                 amsgrad: bool = False,
                 scheduler: str = "exponential",
                 weight_decay: float = 1e-7,
                 lr_decay: float = 0.9,
                 burn_in: int = 0,
                 pooling: str = "max",
                 nonlinearity: str = "elu",
                 fix_param: bool = True,
                 mask: bool = True,
                 **kwargs
                 ):
        super().__init__()
        if learning_rate == "auto":
            # doesn't matter what we set here, this is only used for lr-find
            learning_rate = 1e-3

        self.save_hyperparameters()

        self.train_error = pl.metrics.Accuracy()
        self.valid_error = pl.metrics.Accuracy()
        self.test_error = pl.metrics.Accuracy()

    @abstractmethod
    def forward(x):
        pass

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and (not hasattr(self, "grid") or self.grid is None):
            self.log("equi_error/train", self._equivariance_error(batch))
        loss, y_hat, y = self.shared_step(batch, batch_idx)
        self.log("loss/train", loss)
        self.log("error/train", 100. * (1. - self.train_error(y_hat, y)))
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and (not hasattr(self, "grid") or self.grid is None):
            self.log("equi_error/val", self._equivariance_error(batch))
        loss, y_hat, y = self.shared_step(batch, batch_idx)
        self.log("loss/val", loss)
        self.log("error/val", 100. * (1. - self.valid_error(y_hat, y)), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0 and (not hasattr(self, "grid") or self.grid is None):
            self.log("equi_error/test", self._equivariance_error(batch))
        loss, y_hat, y = self.shared_step(batch, batch_idx)
        self.log("loss/test", loss)
        self.log("error/test", 100. * (1. - self.test_error(y_hat, y)), on_step=False, on_epoch=True)
    
    def shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, y)
        return loss, pred, y
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay,
                                     amsgrad=self.hparams.amsgrad)
        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay,
                                     amsgrad=self.hparams.amsgrad)
        elif self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay,
                                    dampening=0, momentum=0.9, nesterov=True)
        elif self.hparams.optimizer == "sfcnn":
            optim = build_optimizer_sfcnn(self, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("optimizer must be adam or sfcnn")

        if self.hparams.scheduler == "exponential":
            # we decay the learning rate exponentially after a burn-in phase
            def lambda_lr(epoch):
                if epoch < self.hparams.burn_in:
                    return 1
                return self.hparams.lr_decay ** (epoch - self.hparams.burn_in + 1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr)
        else:
            raise ValueError(f"lr scheduler {self.hparams.scheduler} not recognized")
        return [optim], [scheduler]
    
    def _equivariance_error(self, batch, N = 8):
        x, _ = batch
        # there's no need that this is the same group that the actual
        # network implements. It just determines, which group we use to test
        # (since the input is trivial anyway)
        r2_act = gspaces.Rot2dOnR2(N=N)
        in_type = nn.FieldType(r2_act, [r2_act.trivial_repr] * self.hparams.in_channels)
        out1 = self(x)
        errors = torch.empty(N)
        for i, g in enumerate(r2_act.testing_elements):
            with torch.no_grad():
                # Transferring to cpu is necessary because .transform(g) internally
                # uses numpy. Not ideal but this isn't called often.
                x_transformed = in_type.transform(x.cpu(), g).to(self.device)
                out2 = self(x_transformed)
                std = (out1.std(dim=1)**2 + out2.std(dim=1)**2).sqrt()
                batch_errors = ((out1 - out2)**2).sum(dim=1).sqrt() / std
                errors[i] = batch_errors.mean() / self.hparams.n_classes
        return errors.mean()


class EquivariantNetwork(Network):
    def __init__(self,
                 group_order: int = 16,
                 n_classes: int = 10,
                 input_size: int = 28,
                 in_channels: int = 1,
                 learning_rate: float = 1e-3,
                 layer_types: Union[List[str], str] = "kernel",
                 channels: List[int] = [16, 24, 32, 32, 48, 64],
                 kernel_size: List[int] = [7, 5, 5, 5, 5, 5],
                 padding: List[int] = [1, 2, 2, 2, 2, 0],
                 pool_positions: List[int] = [1, 2],
                 # None means that pooling should be used at the end
                 # Otherwise, this should be the spatial size after
                 # the final convolutional layer
                 final_size: int = None,
                 fc_hidden: List[int] = [64],
                 conv_dropout: float = 0.0,
                 fc_dropout: float = 0.0,
                 optimizer: str = "adam",
                 amsgrad: bool = False,
                 weight_decay: float = 1e-7,
                 lr_decay: float = 0.9,
                 burn_in: int = 0,
                 pooling: str = "max",
                 nonlinearity: str = "elu",
                 fix_param: bool = True,
                 mask: bool = True,
                 maximum_offset: int = None,
                 init: Union[str, Dict[str, str]] = "he",
                 maximum_order: int = None,
                 maximum_partial_order: int = None,
                 maximum_power: int = None,
                 special_regular_basis: bool = False,
                 accuracy: int = None,
                 rbffd: bool = False,
                 smoothing: Union[float, List[float]] = None,
                 restrict_kernel_size: bool = False,
                 angle_offset: float = None,
                 normalize_basis: bool = True,
                 rotate_basis: bool = False,
                 flip: bool = False,
                 batch_norm_momentum: float = 0.1,
                 batch_norm_epsilon: float = 1e-5,
                 fc_batch_norm: bool = True,
                 bias: bool = True,
                 restriction_layer: int = None,
                 quotient: bool = False,
                 **kwargs
                 ):
        super().__init__(
            group_order=group_order,
            n_classes=n_classes,
            input_size=input_size,
            in_channels=in_channels,
            learning_rate=learning_rate,
            layer_types=layer_types,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            pool_positions=pool_positions,
            final_size=final_size,
            fc_hidden=fc_hidden,
            conv_dropout=conv_dropout,
            fc_dropout=fc_dropout,
            optimizer=optimizer,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
            lr_decay=lr_decay,
            burn_in=burn_in,
            pooling=pooling,
            nonlinearity=nonlinearity,
            fix_param=fix_param,
            mask=mask,
            maximum_offset=maximum_offset,
            init=init,
            maximum_order=maximum_order,
            maximum_partial_order=maximum_partial_order,
            maximum_power=maximum_power,
            special_regular_basis=special_regular_basis,
            accuracy=accuracy,
            rbffd=rbffd,
            smoothing=smoothing,
            restrict_kernel_size=restrict_kernel_size,
            angle_offset=angle_offset,
            normalize_basis=normalize_basis,
            rotate_basis=rotate_basis,
            flip=flip,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon,
            fc_batch_norm=fc_batch_norm,
            bias=bias,
            restriction_layer=restriction_layer,
            quotient=quotient,
            **kwargs
        )

        if isinstance(layer_types, str):
            layer_types = [layer_types] * len(channels)
        if isinstance(smoothing, (float, int)) or smoothing is None:
            smoothing = [smoothing] * len(channels)

        if pooling == "max":
            create_pooling = lambda in_type: nn.PointwiseMaxPool(in_type, kernel_size=2)
        elif pooling == "avg":
            create_pooling = lambda in_type: nn.PointwiseAvgPoolAntialiased(in_type, sigma=0.66, stride=2)
        else:
            raise ValueError("Pooling must be max or avg")
        
        if nonlinearity == "elu":
            create_nonlinearity = lambda in_type: nn.ELU(in_type, inplace=True)
        elif nonlinearity == "relu":
            create_nonlinearity = lambda in_type: nn.ReLU(in_type, inplace=True)
        else:
            raise ValueError("Non-linearity must be elu or relu")
        
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=group_order)
        elif group_order == 1:
            self.r2_act = gspaces.TrivialOnR2()
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=group_order)

        assert len(kernel_size) == len(channels)
        assert len(padding) == len(channels)
        
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * in_channels)
        
        self.input_type = in_type

        layers = []
        
        for i in range(len(channels)):
            if fix_param:
                # to keep number of parameters more or less constant when changing groups
                # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
                t = self.r2_act.fibergroup.order() / 16
                C = int(round(channels[i] / np.sqrt(t)))
            else:
                C = channels[i]
            
            if group_order == 1 and not flip:
                repr = self.r2_act.trivial_repr
                out_type = nn.FieldType(self.r2_act, C * [repr])
            elif not self.hparams.quotient:
                repr = self.r2_act.regular_repr
                out_type = nn.FieldType(self.r2_act, C * [repr])
            else:
                assert self.hparams.group_order == 16
                assert not self.hparams.flip
                repr = [self.r2_act.regular_repr] * 5
                repr += [self.r2_act.quotient_repr(2)] * 2
                repr += [self.r2_act.quotient_repr(4)] * 2
                repr += [self.r2_act.trivial_repr] * 4

                C /= sum([r.size for r in repr]) / self.r2_act.fibergroup.order()
                C = int(round(C))
            
                out_type = nn.FieldType(self.r2_act, repr * C).sorted()
            block = []
            if i == 0 and mask:
                block.append(nn.MaskModule(in_type, self.hparams.input_size, margin=1))
            block.append(self._create_layer(
                layer_types[i],
                in_type,
                out_type,
                kernel_size=kernel_size[i],
                padding=padding[i],
                smoothing=smoothing[i],
                bias=bias,
            ))
            if i + 1 == self.hparams.restriction_layer:
                assert self.hparams.flip, "Restriction is only supported from D_N to C_N"
                block.append(nn.RestrictionModule(out_type, (None, self.hparams.group_order)))
                block.append(nn.DisentangleModule(block[-1].out_type))
                out_type = block[-1].out_type
                self.r2_act, _, _ = self.r2_act.restrict((None, self.hparams.group_order))
            block.append(nn.InnerBatchNorm(out_type, momentum=batch_norm_momentum, eps=batch_norm_epsilon))
            block.append(create_nonlinearity(out_type))
            if self.hparams.conv_dropout > 0 and i not in pool_positions:
                block.append(nn.PointwiseDropout(out_type, p=self.hparams.conv_dropout))
            layers.append(nn.SequentialModule(*block))

            in_type = out_type

            if i in pool_positions:
                layers.append(create_pooling(out_type))
        
        if final_size is None:
            layers.append(nn.GroupPooling(in_type))
            out_type = layers[-1].out_type

            # pool spatial dimensions to scalar
            if pooling == "max":
                layers.append(nn.PointwiseAdaptiveMaxPool(out_type, 1))
            elif pooling == "avg":
                layers.append(nn.PointwiseAdaptiveAvgPool(out_type, 1))
            
            final_size = 1

        self.equi_net = nn.SequentialModule(*layers)
        # number of output channels
        c = self.equi_net.out_type.size * final_size ** 2

        # Fully Connected
        # HACK: an empty list is passed in as a ListConfig object by Hydra, apparently,
        # so we need to convert it
        if not isinstance(fc_hidden, list):
            fc_hidden = list(fc_hidden)
        ins = [c] + fc_hidden
        outs = fc_hidden + [n_classes]
        fc_layers = []
        for i in range(len(ins)):
            # NOTE: in the SFCNN architecture from the e2cnn paper implementation,
            # dropout is placed here, before the linear layer, so for now we
            # copy that
            if self.hparams.fc_dropout > 0:
                fc_layers.append(torch.nn.Dropout(p=self.hparams.fc_dropout))

            fc_layers.append(torch.nn.Linear(ins[i], outs[i], bias=bias))
            if fc_batch_norm:
                fc_layers.append(torch.nn.BatchNorm1d(outs[i], momentum=batch_norm_momentum, eps=batch_norm_epsilon))
            if i < len(ins) - 1:
                fc_layers.append(torch.nn.ELU(inplace=True))

        self.fully_net = torch.nn.Sequential(*fc_layers)

    def _create_layer(self,
                      method: str,
                      in_type: nn.FieldType,
                      out_type: nn.FieldType,
                      kernel_size: int,
                      padding: int,
                      smoothing: float,
                      bias: bool = True) -> nn.EquivariantModule:
        if isinstance(self.hparams.init, Mapping):
            init = self.hparams.init[method]
        else:
            init = self.hparams.init

        if method == "kernel":
            return nn.R2Conv(
                in_type, out_type,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                maximum_offset=self.hparams.maximum_offset,
                init=init,
            )
        elif method == "diffop":
            return nn.R2Diffop(
                in_type,
                out_type,
                accuracy=self.hparams.accuracy,
                maximum_order=self.hparams.maximum_order,
                maximum_partial_order=self.hparams.maximum_partial_order,
                maximum_offset=self.hparams.maximum_offset,
                maximum_power=self.hparams.maximum_power,
                special_regular_basis=self.hparams.special_regular_basis,
                cache=False,
                rbffd=self.hparams.rbffd,
                kernel_size=kernel_size,
                padding=padding,
                init=init,
                bias=bias,
                smoothing=smoothing,
                restrict_kernel_size=self.hparams.restrict_kernel_size,
                normalize_basis=self.hparams.normalize_basis,
                angle_offset=self.hparams.angle_offset,
                rotate90=self.hparams.rotate_basis,
            )
        else:
            raise ValueError(f"Method must be 'kernel' or 'diffop', got {method}")

    def forward(self, x):
        x = nn.GeometricTensor(x, self.input_type)
        x = self.equi_net(x)
        x = x.tensor
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x


class CNN(Network):
    def __init__(self,
                 n_classes: int = 10,
                 learning_rate: float = 1e-3,
                 channels: List[int] = [16, 24, 32, 32, 48, 64],
                 kernel_size: List[int] = [7, 5, 5, 5, 5, 5],
                 padding: List[int] = [1, 2, 2, 2, 2, 0],
                 pool_positions: List[int] = [1, 2],
                 final_size: int = None,
                 fc_hidden: List[int] = [64],
                 fc_dropout: float = 0.0,
                 optimizer: str = "adam",
                 weight_decay: float = 1e-7,
                 lr_decay: float = 0.9,
                 burn_in: int = 0,
                 pooling: str = "max",
                 nonlinearity: str = "elu",
                 fix_param: bool = True,
                 mask: bool = True,
                 maximum_offset: int = None,
                 **kwargs
                 ):
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            pool_positions=pool_positions,
            final_size=final_size,
            fc_hidden=fc_hidden,
            fc_dropout=fc_dropout,
            optimizer=optimizer,
            weight_decay=weight_decay,
            lr_decay=lr_decay,
            burn_in=burn_in,
            pooling=pooling,
            nonlinearity=nonlinearity,
            fix_param=fix_param,
            mask=mask,
            maximum_offset=maximum_offset,
            **kwargs
        )

        if pooling == "max":
            create_pooling = lambda: torch.nn.MaxPool2d(2)
        elif pooling == "avg":
            create_pooling = lambda: torch.nn.AvgPool2d(2)
        else:
            raise ValueError("Pooling must be max or avg")
        
        if nonlinearity == "elu":
            create_nonlinearity = lambda: torch.nn.ELU(inplace=True)
        elif nonlinearity == "relu":
            create_nonlinearity = lambda: torch.nn.ReLU(inplace=True)
        else:
            raise ValueError("Non-linearity must be elu or relu")
        
        if fix_param:
            # HACK: This is currently just hard-coded to reach roughly the same parameter
            # number as the C_16 steerable CNN with default settings.
            # The issue is that this depends on the kernel size and cutoff that
            # is used in the steerable CNNs.
            self._width_adjust = 1/2.5
        else:
            self._width_adjust = 1
        
        channels = [round(c / self._width_adjust) for c in channels]
        assert len(kernel_size) == len(channels)
        assert len(padding) == len(channels)
        
        layers = []
        ins = [1] + channels[:-1]
        outs = channels


        for i in range(len(channels)):
            layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(ins[i], outs[i], kernel_size[i], padding=padding[i]),
                torch.nn.BatchNorm2d(outs[i]),
                create_nonlinearity()
            ))

            if i in pool_positions:
                layers.append(create_pooling())

        if final_size is None:
            # pool spatial dimensions to scalar
            if pooling == "max":
                layers.append(torch.nn.AdaptiveMaxPool2d(1))
            elif pooling == "avg":
                layers.append(torch.nn.AdaptiveAvgPool2d(1))
            
            final_size = 1

        self.conv_net = torch.nn.Sequential(*layers)
        # number of output channels
        c = outs[-1] * final_size ** 2

        # Fully Connected
        # HACK: an empty list is passed in as a ListConfig object by Hydra, apparently,
        # so we need to convert it
        if not isinstance(fc_hidden, list):
            fc_hidden = list(fc_hidden)
        ins = [c] + fc_hidden
        outs = fc_hidden + [n_classes]
        fc_layers = []
        for i in range(len(ins)):
            # NOTE: in the SFCNN architecture from the e2cnn paper implementation,
            # dropout is placed here, before the linear layer, so for now we
            # copy that
            if self.hparams.fc_dropout > 0:
                fc_layers.append(torch.nn.Dropout(p=self.hparams.fc_dropout))

            fc_layers.append(torch.nn.Linear(ins[i], outs[i]))
            fc_layers.append(torch.nn.BatchNorm1d(outs[i]))
            if i < len(ins) - 1:
                fc_layers.append(torch.nn.ELU(inplace=True))

        self.fully_net = torch.nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x

