# Winner-Take-All Ensemble Neural Network
# Copyright (C) 2021  Abien Fred Agarap
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Implementation of kWTA-ENN"""
from typing import Dict, Tuple

import torch

from kwta_ensemble.models.base import Model


class kWTAEnsemble(Model):
    def __init__(
        self,
        num_classes: int,
        expert_model: torch.nn.Module,
        num_experts: int = 2,
        sparsity: float = 0.75,
        competition_delay: int = 3,
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass

    def epoch_train(self, data_loaders: Dict, phase: str) -> Tuple[float, float]:
        pass
