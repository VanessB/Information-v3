# Experiments

This folder consists of several Jupyter notebooks with MI estimation experiments.

List of conducted experiments with synthetic data:
- Trivial tests (normal and uniform distributions) [link](<./No compression tests.ipynb>)
- Rasterised images of 2D-gaussians (diagonal covariance matrix): [AE](<./Gaussian plots, AE.ipynb>), [PCA](<./Gaussian plots, PCA.ipynb>)
- Rasterised images of 2D-gaussians: [AE](<./Correlated Gaussian plots, AE.ipynb>)
- Rasterised images of rectangles: [AE](<./Rectangles, AE.ipynb>), [PCA](<./Rectangles, AE.ipynb>)
- Highly nonlinear manifold example: [link](<./Anti-PCA.ipynb>)

List of conducted experiments with DNNs:
- DNN classifier of MNIST dataset: [link](<./Information plane experiments (DNN classifier, MNIST).ipynb>)
- DNN classifier of CIFAR10 dataset: [link](<./Information plane experiments (DNN classifier, CIFAR10).ipynb>)

Necessary utils and NN architectures can be found in the [`misc` directory](./misc).