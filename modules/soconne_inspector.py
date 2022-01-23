# k-Winners-Take-All Ensemble Neural Network
# Copyright (C) 2021  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Inspector module for kWTA Ensemble"""
import sys
from typing import List

import torch
from prefex.models.snnl_ae import Autoencoder
from pt_datasets import create_dataloader, load_dataset
from sklearn.metrics import classification_report

from kwta_ensemble.models import DNN, kWTAEnsemble
from kwta_ensemble.utils import (
    compute_learner_accuracy,
    compute_learner_accuracy_per_class,
    compute_learner_classification_report,
    display_accuracies,
    display_learner_accuracy_per_class,
    display_reports,
    plot_activations,
    set_global_seed,
)


def compute_expert_wta_outputs(
    model: torch.nn.Module, expert_outputs: List, num_data: int
) -> List:
    """
    Computes the kWTA outputs of each expert in `model`.

    Parameters
    ----------
    model: torch.nn.Module
        The kWTA ensemble model.
    expert_outputs: List
        The individual expert outputs.
    num_data: int
        The number of examples to create.

    Returns
    -------
    expert_logits: List
        The kWTA outputs of experts.
    """
    num_classes = 10
    expert_logits = list()
    for index in range(len(expert_outputs)):
        zeros = torch.zeros(num_data, (model.num_subnetworks * num_classes))
        offset = index * num_classes
        zeros[:, offset : offset + num_classes] = expert_outputs[index]
        logits = model.competitive_layer(zeros)
        expert_logits.append(logits)
    return expert_logits


filename = sys.argv[1]
use_snnl = sys.argv[2]
seed = int(filename.split("-", 6)[0])
model = filename.split("-", 6)[2]
num_learners = int(filename.split("-", 6)[3])
learners_arch = filename.split("-", 6)[4]
dataset = filename.split("-", 6)[5]

set_global_seed(seed)
_, test_data = load_dataset(dataset)
test_loader = create_dataloader(test_data, batch_size=len(test_data), shuffle=False)

encoder = Autoencoder(
    num_features=784,
    code_dim=50,
    criterion="bce",
    optimizer="adamw",
    learning_rate=1e-3,
    use_lr_scheduling=True,
    use_snnl=(True if use_snnl == "True" else False),
    temperature=1.0,
    factor=1.0,
    mode="autoencoding",
    code_units=50,
)
learner = DNN(units=((50, 100), (100, 10)))
model = kWTAEnsemble(
    expert_model=learner,
    num_subnetworks=num_learners,
    num_classes=10,
    sparsity=0.75,
    competition_delay=0,
    optimizer="sgd",
    learning_rate=1e-3,
    weight_decay=1e-5,
    use_feature_extractor=True,
    feature_extractor=encoder.layers[:8],
)
model.load_model(filename)
model = model.cpu()

accuracy = model.score(test_loader)
for features, labels in test_loader:
    outputs = model.predict(features)

report = classification_report(outputs.argmax(1).detach().numpy(), labels.numpy())

encoded_features = model.feature_extractor(features.view(features.shape[0], -1))
expert_outputs = list(map(lambda expert: expert(encoded_features), model.experts))
expert_logits = compute_expert_wta_outputs(model, expert_outputs, len(test_data))
expert_accuracies = compute_learner_accuracy(outputs=expert_logits, labels=labels)
expert_reports = compute_learner_classification_report(
    outputs=expert_logits, labels=labels
)
expert_class_accuracies = compute_learner_accuracy_per_class(
    outputs=[outputs, *expert_logits], labels=labels
)
expert_outputs = list(
    map(lambda output: torch.nn.functional.softmax(output), expert_logits)
)

outputs = torch.nn.functional.softmax(outputs)

index = int(sys.argv[3])
classes = test_data.classes
if dataset == "mnist":
    classes = list(map(lambda index: str(index), range(10)))
plot_activations(
    index=index,
    features=features,
    labels=labels,
    classes=classes,
    outputs=expert_outputs,
    model_output=outputs,
)

display_accuracies(accuracies=[accuracy, *expert_accuracies])
display_reports(reports=[report, *expert_reports])
display_learner_accuracy_per_class(accuracies=expert_class_accuracies)
