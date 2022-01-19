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
"""kWTA Ensemble classifier"""
import argparse

import numpy as np
from prefex.models.classifier import Prefex
from prefex.models.snnl_ae import Autoencoder
from prefex.models.supervised_ae import SupervisedAutoencoder
from soconne_baseline import ResNet18, ResNet34, ResNet50

from kwta_ensemble.models import CNN, DNN, LeNet, PrefexDNN, kWTAEnsemble
from kwta_ensemble.utils import (
    create_dataloaders,
    export_results,
    get_kwta_enn_filename,
    set_global_seed,
)


def main(arguments: argparse.Namespace):
    (
        seeds,
        dataset,
        vectorizer,
        ngram_range,
        batch_size,
        optimizer,
        learning_rate,
        weight_decay,
        epochs,
        num_subnetworks,
        subnetwork_architecture,
        show_every,
        competition_delay,
        sparsity_factor,
        use_pretrained_cifar10,
        num_blocks_to_freeze,
        prefex_path,
        use_snnl,
        use_feature_extractor,
    ) = (
        arguments.seeds,
        arguments.dataset,
        arguments.vectorizer,
        arguments.ngram_range,
        arguments.batch_size,
        arguments.optimizer,
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.epochs,
        arguments.num_subnetworks,
        arguments.subnetwork_architecture,
        arguments.show_every,
        arguments.use_competition_after,
        arguments.sparsity_factor,
        arguments.use_pretrained_cifar10,
        arguments.num_blocks_to_freeze,
        arguments.prefex_path,
        arguments.use_snnl,
        arguments.use_feature_extractor,
    )
    results = dict()
    for num_subnetwork in range(2, num_subnetworks + 1):
        accuracies = []
        for seed in seeds:
            print()
            print(f"[INFO] Dataset: {dataset}")
            print(f"[INFO] Number of learners: {num_subnetwork}")
            print(f"[INFO] Seed: {seed}")

            set_global_seed(seed=seed)

            data_loaders = create_dataloaders(
                dataset=dataset,
                vectorizer=vectorizer,
                ngram_range=ngram_range,
                batch_size=batch_size,
                seed=seed,
            )

            train_loader = data_loaders.get("train")
            valid_loader = data_loaders.get("valid")
            test_loader = data_loaders.get("test")
            num_features = data_loaders.get("meta").get("num_features")
            input_shape = data_loaders.get("meta").get("input_shape")
            num_classes = data_loaders.get("meta").get("num_classes")

            if subnetwork_architecture == "dnn":
                if use_feature_extractor:
                    # encoder = Autoencoder(
                    #     code_dim=50,
                    #     criterion="bce",
                    #     learning_rate=1e-2,
                    #     optimizer="adamw",
                    # )
                    # print(encoder)
                    # encoder = Prefex(
                    #     encoder=encoder.encoder,
                    #     learning_rate=1e-1,
                    #     use_snnl=use_snnl,
                    #     temperature=10.0,
                    #     factor=-10.0
                    # )
                    # print(encoder)
                    # encoder = SupervisedAutoencoder(
                    #     code_dim=50,
                    #     criterion="bce",
                    #     optimizer="adamw",
                    #     learning_rate=1e-1,
                    #     use_snnl=use_snnl,
                    #     temperature=10.0,
                    #     factor=-10.0,
                    # )
                    encoder = Autoencoder(
                        num_features=num_features,
                        code_dim=200,
                        criterion="bce",
                        optimizer="adamw",
                        learning_rate=1e-3,
                        use_lr_scheduling=True,
                        use_snnl=use_snnl,
                        temperature=100.0,
                        factor=100.0,
                        mode="latent_code",
                        code_units=int(0.70 * 200),
                    )
                    encoder.load_model(prefex_path)
                    subnetwork = DNN(units=((200, 100), (100, num_classes)))
                else:
                    subnetwork = DNN(units=((num_features, 100), (100, num_classes)))
            elif subnetwork_architecture == "cnn":
                subnetwork = CNN(
                    dim=input_shape[1],
                    input_dim=(1 if len(input_shape) < 4 else input_shape[3]),
                    num_classes=num_classes,
                )
            elif subnetwork_architecture == "lenet":
                subnetwork = LeNet(
                    dim=input_shape[1],
                    channel_dim=(1 if len(input_shape) < 4 else input_shape[3]),
                    num_classes=num_classes,
                )
            elif subnetwork_architecture == "resnet18":
                subnetwork = ResNet18(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                    blocks_to_freeze=num_blocks_to_freeze,
                    use_pretrained_cifar10=use_pretrained_cifar10,
                )
            elif subnetwork_architecture == "resnet34":
                subnetwork = ResNet34(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                    blocks_to_freeze=num_blocks_to_freeze,
                    use_pretrained_cifar10=use_pretrained_cifar10,
                )
            elif subnetwork_architecture == "resnet50":
                subnetwork = ResNet50(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=learning_rate,
                    blocks_to_freeze=num_blocks_to_freeze,
                    use_pretrained_cifar10=use_pretrained_cifar10,
                )

            model = kWTAEnsemble(
                num_classes=num_classes,
                expert_model=subnetwork,
                num_subnetworks=num_subnetwork,
                sparsity=sparsity_factor,
                competition_delay=competition_delay,
                optimizer=optimizer,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                use_feature_extractor=use_feature_extractor,
                # feature_extractor=encoder.layers[:-2],
                # feature_extractor=encoder.encoder[:-1],
                feature_extractor=encoder.layers[:7]
            )
            from torchsummary import summary

            print(summary(model, (1, 784)))
            model.fit(train_loader, valid_loader, epochs=epochs, show_every=show_every)
            accuracy = model.score(test_loader)
            accuracies.append(accuracy)
            results[f"acc_seed_{seed}"] = accuracy
            results[f"training_loss_{seed}"] = model.train_loss
            results[f"training_acc_{seed}"] = model.train_accuracy
            results[f"valid_loss_{seed}"] = model.valid_loss
            results[f"valid_acc_{seed}"] = model.valid_accuracy

            print(f"Test acc: {accuracy:.4f}")
            filename = get_kwta_enn_filename(
                num_subnetwork=num_subnetwork,
                subnetwork_architecture=subnetwork_architecture,
                dataset=dataset,
                learning_rate=learning_rate,
                optimizer=optimizer,
                batch_size=batch_size,
                competition_delay=competition_delay,
                sparsity_factor=sparsity_factor,
            )
            model.save_model(filename=f"{seed}-seed-{filename}")
        print()
        print("=" * 40)
        print(f"AVG ACC = {np.mean(accuracies):.4f}")
        print(f"MAX ACC = {np.max(accuracies):.4f}")
        print(f"STDDEV ACC = {np.std(accuracies):.4f}")
        print("=" * 40)
        results["acc_avg"] = np.mean(accuracies)
        results["acc_max"] = np.max(accuracies)
        results["acc_std"] = np.std(accuracies)
        export_results(model_results=results, filename=filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description="kWTA Ensemble Neural Network classifier"
    )
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-s",
        "--seeds",
        nargs="+",
        type=int,
        required=False,
        default=[1234, 42, 73],
        help="the list of random seeds to use, default: [1234, 42, 73]",
    )
    group.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        help="the dataset to use, default: [mnist]",
    )
    group.add_argument(
        "-v",
        "--vectorizer",
        type=str,
        default="ngrams",
        help="the vectorizer to use, options: [ngrams (default) | tfidf]",
    )
    group.add_argument(
        "-nr",
        "--ngram_range",
        nargs="+",
        required=False,
        default=(1, 5),
        type=int,
        help="the n-grams range to use for text vectorization, default: [(1, 5)]",
    )
    group.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="the mini-batch size to use, default: [256]",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="the optimization algorithm to use, options: [sgd (default) | adamw]",
    )
    group.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3e-4,
        help="the learning rate to use for optimization, default: [3e-4]",
    )
    group.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-5,
        help="the weight decay to use during optimization, default: [1e-5]",
    )
    group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="the number of epochs to train the model, default: [10]",
    )
    group.add_argument(
        "-se",
        "--show_every",
        type=int,
        default=1,
        help="the number of interval between training progress displays, default: [1]",
    )
    group.add_argument(
        "-nl",
        "--num_subnetworks",
        type=int,
        default=3,
        help="the number of subnetworks to instantiate, default: [3]",
    )
    group.add_argument(
        "--subnetwork_architecture",
        type=str,
        default="dnn",
        help="the architecture to use for an expert, options: [cnn | dnn (default) | lenet | prefex_dnn]",
    )
    group.add_argument(
        "-cd",
        "--use_competition_after",
        type=int,
        default=0,
        help="the number of epochs to reach before using kWTA, default: [0]",
    )
    group.add_argument(
        "-sf",
        "--sparsity_factor",
        type=float,
        default=0.75,
        help="the percentage of winners to get, default: [0.75]",
    )
    group.add_argument(
        "--use_pretrained_cifar10",
        required=False,
        dest="use_pretrained_cifar10",
        action="store_true",
    )
    group.add_argument(
        "--num_blocks_to_freeze",
        type=int,
        default=4,
        help="the number of ResNet blocks to freeze, default: [4]",
    )
    group.add_argument(
        "--prefex_path", type=str, help="the path to the pretrained feature extractor."
    )
    group.add_argument(
        "--use_snnl", required=False, dest="use_snnl", action="store_true"
    )
    group.add_argument(
        "--use_feature_extractor",
        required=False,
        dest="use_feature_extractor",
        action="store_true",
    )
    group.set_defaults(use_pretrained_cifar10=False)
    group.set_defaults(use_snnl=False)
    group.set_defaults(use_feature_extractor=False)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
