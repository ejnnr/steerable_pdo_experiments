from e2cnn import nn
import torch
from torchvision.transforms.functional import rotate
import numpy as np

# Based on
# https://github.com/QUVA-Lab/e2cnn_experiments/blob/0c8f275be0361367c52d2d268471ac32f39fe3f3/experiments/experiment.py
#
# Original license:
# 
# Copyright (c) 2021 Qualcomm Innovation Center, Inc. 
# All rights reserved. 
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the 
# disclaimer below) provided that the following conditions are met: 
#
# * Redistributions of source code must retain the above copyright 
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright 
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution. 
#
# * Neither the name of Qualcomm Innovation Center, Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission. 
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
# GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
# HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def build_optimizer_sfcnn(model, lr, weight_decay):
    # optimizer as in "Learning Steerable Filters for Rotation Equivariant CNNs"
    # https://arxiv.org/abs/1711.07289
    
    # split up parameters into groups, named_parameters() returns tuples ('name', parameter)
    # each group gets its own regularization gain
    batchnormLayers = [m for m in model.modules() if isinstance(m, (
        nn.NormBatchNorm,
        nn.GNormBatchNorm,
        nn.InnerBatchNorm,
        nn.InducedNormBatchNorm,
    ))]
    linearLayers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    convlayers = [m for m in model.modules() if isinstance(m, (
        torch.nn.Conv2d,
        nn.R2Layer,
    ))]
    weights_conv = [p for m in convlayers for n, p in m.named_parameters() if n.endswith('weights') or n.endswith("weight")]
    biases = [p for n, p in model.named_parameters() if n.endswith('bias')]
    weights_bn = [p for m in batchnormLayers for n, p in m.named_parameters()
                  if n.endswith('weight') or n.split('.')[-1].startswith('weight')
                  ]
    weights_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('weight')]
    # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    print("SFCNN optimizer")
    for n, p in model.named_parameters():
        if p.requires_grad and not n.endswith(('weight', 'weights', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(n))
    param_groups = [
        dict(params=weights_conv, weight_decay=weight_decay),
        dict(params=weights_bn, weight_decay=0),
        dict(params=weights_fully, weight_decay=weight_decay),
        dict(params=weights_softmax, weight_decay=0),
        dict(params=biases, weight_decay=0)
    ]
    return torch.optim.Adam(param_groups, lr=lr, betas=(0.9, 0.999))


class ZeroRotation(torch.nn.Module):
    def __init__(self, interpolation):
        super().__init__()
        self.interpolation = interpolation
    
    def forward(self, x):
        return rotate(x, 0, self.interpolation)

