# k-Winners-Take-All Ensemble Neural Network
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
"""Inspector for MoE classifier outputs"""
import sys

from moe.models import MoE
from pt_datasets import create_dataloader, load_dataset
from sklearn.metrics import classification_report
import torch

from kwta_ensemble.models import DNN
from kwta_ensemble.utils import (
    compute_learner_accuracy,
    compute_learner_classification_report,
    set_global_seed,
)


filename = sys.argv[1]
seed = int(filename.split("-", 6)[0])
model = filename.split("-", 6)[2]
num_learners = int(filename.split("-", 6)[3])
learners_arch = filename.split("-", 6)[4]
dataset = filename.split("-", 6)[5]


set_global_seed(seed)
_, test_data = load_dataset(dataset)
test_loader = create_dataloader(test_data, batch_size=len(test_data), shuffle=False)

expert = DNN(units=((784, 100), (100, 10)))
gating = DNN(units=((784, 100), (100, num_learners)))
model = MoE(expert_model=expert, gating_model=gating, num_experts=num_learners)
model.load_model(filename)
model = model.cpu()

accuracy = model.score(test_loader)
for features, labels in test_loader:
    outputs = model.predict(features)

report = classification_report(outputs.argmax(1).detach().numpy(), labels.numpy())

expert_outputs = list(map(lambda expert: expert(features), model.experts))
expert_outputs = list(
    map(
        lambda expert_output: torch.nn.functional.softmax(expert_output), expert_outputs
    )
)
expert_accuracies = compute_learner_accuracy(outputs=expert_outputs, labels=labels)
expert_reports = compute_learner_classification_report(
    outputs=expert_outputs, labels=labels
)
