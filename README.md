# Mutinfo
[Russian/Русский](./README_ru.md)

An information-theoretic framework to study datasets and neural networks.

## Features
- Kernel Density Estimation (maximum likelihood and LSE) mutual information estimators.
- Kozachenko-Leonenko (original and weighted) mutual information estimators.
- Framework for mutual information estimation via lossy compression.
- Synthetic datasets with predefined information-theoretic quantities.
- Information bottleneck experiments with neural networks.

## Structure

- `/source/python/mutinfo` — source code of the framework, including submodules for synthetic dataset generation.
- `/source/examples` — `.ipynb` files to demonstrate the framework and conduct experiments.
- `/source/gnuplot` — gnuplot scripts to plot data acquired from experiments.