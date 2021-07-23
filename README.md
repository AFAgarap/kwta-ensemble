# k-Winners-Take-All Ensemble Neural Network

## Abstract

Ensembling is one approach that improves the performance of a neural network by
combining a group of them. It combines a number of independent neural networks
usually by averaging or summing up their individual outputs. In this work, we
modify the ensemble of independent neural networks by training the sub-networks
concurrently instead of independently. This concurrent training of sub-networks
lead them to cooperate with each other, and we refer to them as "cooperative
ensemble" of networks. Meanwhile, the mixture-of-experts approach improves the
performance of a neural network by dividing up a given dataset to a group of
neural networks; and with a gating network assigning a specialization to each
of its sub-networks called "experts". Instead of the aforementioned ways for
combining a group of neural networks, we propose to use a k-Winners-Take-All
(kWTA) activation function to act as the combination method for the outputs of
each sub-network in the ensemble, which we call "kWTA ensemble neural
networks" (kWTA-ENN). With kWTA, the losing neurons of the sub-networks are
inhibited while the winning neurons are retained, which results to sub-networks
having some form of specialization but also sharing knowledge with one another.
We compare our approach with the cooperative ensemble of networks and
mixture-of-experts, where we used a feed-forward network with one hidden layer
having 100 neurons as the sub-network architecture. Our approach yielded a
better performance than our baseline models, reaching the following test
accuracies on benchmark datasets: 98.34% on MNIST, 88.06% on Fashion-MNIST,
91.61% on KMNIST, and 95.97% on WDBC.

## Usage

It is recommended to create a virtual environment for this repository.

```buildoutcfg
$ pyenv virtualenv 3.8.8 kwta-ensemble
$ pyenv local kwta-ensemble
```

Then, either `pip` or `poetry` could be used to set up its dependencies.

```
$ pip install -r requirements
$ # or
$ poetry install
```

The following scripts can be run to reproduce the results below:

**Mixture-of-Experts** on MNIST, Fashion-MNIST, KMNIST, and WDBC.

```buildoutcfg
$ bash scripts/run_moe "mnist" 100 1e-1 5 "dnn" "sgd"
$ bash scripts/run_moe "fashion_mnist" 100 1e-1 5 "dnn" "sgd"
$ bash scripts/run_moe "kmnist" 100 1e-1 5 "dnn" "sgd"
$ bash scripts/run_moe "wdbc" 40 1e-1 5 "dnn" "sgd"
```

**Cooperative Ensemble** on MNIST, Fashion-MNIST, KMNIST, and WDBC.

```buildoutcfg
$ bash scripts/run_ensemble "mnist" 100 1e-1 5 "dnn" "sgd"
$ bash scripts/run_ensemble "fashion_mnist" 100 1e-1 5 "dnn" "sgd"
$ bash scripts/run_ensemble "kmnist" 100 1e-1 5 "dnn" "sgd"
$ bash scripts/run_ensemble "wdbc" 40 1e-1 5 "dnn" "sgd"
```

**kWTA-ENN** on MNIST, Fashion-MNIST, KMNIST, and WDBC.

```buildoutcfg
$ bash scripts/run_kwta "mnist" 100 1e-1 5 "dnn" 0 "sgd"
$ bash scripts/run_kwta "fashion_mnist" 100 1e-1 5 "dnn" 0 "sgd"
$ bash scripts/run_kwta "kmnist" 100 1e-1 5 "dnn" 0 "sgd"
$ bash scripts/run_kwta "wdbc" 100 1e-1 5 "dnn" 0 "sgd"
```

The penultimate parameter in the commands above is the delay parameter.

Note: The full results might not be completely reproduced but rather be
on the same range of performance trend due to software and hardware
configuration of the user. We note the hardware we used for our
experiments below.

## Results

To demonstrate the improvement in model performance using our approach, we used
four benchmark datasets for evaluation: MNIST, Fashion-MNIST, KMNIST,
and Wisconsin Diagnostic Breast Cancer (WDBC). We ran each model ten
times, and we report the average, best, and standard deviation of test
accuracies for each of our model.

**Hardware and Software Configuration** We used a laptop computer withan Intel
Core i5-6300HQ CPU with Nvidia GTX 960M GPU for training all our models. Then we
used the following arbitrarily chosen 10 random seeds for reproducibility: 42,
1234, 73, 1024, 86400, 31415, 2718, 30, 22, and 17.

We evaluate the performance of our proposed approach in its different
configurations as per the competition delay parameter _d_ and compare it with our
baseline models, Mixture-of-Experts (MoE) and Cooperative Ensemble (CE). The
empirical evidence shows our proposed approach outperforms our baseline
models on the benchmark datasets we used. However, we are not able to observe a
trend in performance with respect to the varying values of d, which warrants
further investigation.

![](assets/full-results.png)
Figure 1. Classification results on the benchmark datasets (bold values
represent the best results) in terms of average, best, and standard deviation
of test accuracies (in %). Our k-WTA ensemble network achieves better test
accuracies than our baseline models.

![](assets/mnist-inspection.png)
Figure 2. Predictions of each sub-network on a sample MNIST data and their respective
final outputs. In 2a, we can infer that MoE
sub-networks 2 and 3 are specializing on class 1. In
2b, all CE sub-networks have high
probability outputs for class 1. In 2c,
kWTA-ENN sub-networks 1 and 2 helped each other but with the kWTA function, the
neurons for other classes were most likely inhibited at inference, thus its
higher probability output than MoE and CE.

![](assets/kmnist-inspection.png)
Figure 3. Predictions of each sub-network on a sample KMNIST data and their
respective final outputs. In 3a, we can
infer that MoE sub-network 2 is specializing on class 6 ("ma"). In
3b, CE sub-network 3 was assisted by
sub-network 2. In 3c, kWTA-ENN sub-networks
1 and 2 helped each other but with the kWTA function, the neurons for other
classes were most likely inhibited at inference, thus its higher probability
output than MoE and CE.

![](assets/specialization.png)
Figure 4. Classification results of each kWTA-ENN sub-network and kWTA-ENN
itself on MNIST (4a) and KMNIST
(4b) datasets. The tables show the test accuracy of
each sub-network on each dataset class, indicating a degree of specialization
among the sub-networks. Furthermore, the final model accuracy on each class
show that combining the sub-network outputs have stronger predictive
capability. These divisions were in no way pre-determined but show how
cooperation by specialization can be done through competitive ensemble.

## License

Everything in this repository is intended for non-commercial research purposes
only.

```
k-Winners-Take-All Ensemble Neural Network
Copyright (C) 2021  Abien Fred Agarap

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
