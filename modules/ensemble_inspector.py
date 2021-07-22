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
"""Inspector module for Ensemble outputs"""
import sys

from pt_datasets import create_dataloader, load_dataset
from sklearn.metrics import classification_report
import torch

from kwta_ensemble.models import DNN, Ensemble
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


filename = sys.argv[1]
seed = int(filename.split("-", 6)[0])
model = filename.split("-", 6)[2]
num_learners = int(filename.split("-", 6)[3])
learners_arch = filename.split("-", 6)[4]
dataset = filename.split("-", 6)[5]

set_global_seed(seed)
_, test_data = load_dataset(dataset)
test_loader = create_dataloader(test_data, batch_size=len(test_data), shuffle=False)

learner = DNN(units=((784, 100), (100, 10)))
model = Ensemble(network=learner, num_subnetworks=num_learners)
model.load_model(filename)
model = model.cpu()

accuracy = model.score(test_loader)
for features, labels in test_loader:
    outputs = model.predict(features)

report = classification_report(outputs.argmax(1).detach().numpy(), labels.numpy())

learner_outputs = list(map(lambda learner: learner(features), model.model))
learner_outputs = list(
    map(lambda outputs: torch.nn.functional.softmax(outputs), learner_outputs)
)

learner_accuracies = compute_learner_accuracy(outputs=learner_outputs, labels=labels)
learner_reports = compute_learner_classification_report(
    outputs=learner_outputs, labels=labels
)
learner_class_accuracies = compute_learner_accuracy_per_class(
    outputs=[outputs, *learner_outputs], labels=labels
)

outputs = torch.nn.functional.softmax(outputs)

index = int(sys.argv[2])
plot_activations(
    index=index,
    features=features,
    labels=labels,
    classes=test_data.classes,
    outputs=learner_outputs,
    model_output=outputs,
)

display_accuracies(accuracies=[accuracy, *learner_accuracies])
display_reports(reports=[report, *learner_reports])
display_learner_accuracy_per_class(accuracies=learner_class_accuracies)
