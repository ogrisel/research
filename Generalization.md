# Generalization

- Study Fisher-Rao norm of various initialization schemes.

- Compare Fisher-Rao norm of ResNet vs pure MLP reparametrization at each epoch when fitting on MNIST.

- Compute cosine distance of natural gradient direction with (SGD, SGD + Nesterov, Adam, K-FAC) step at each epoch when fitting ResNet, DenseNet and VGG on CIFAR-10- Study Fisher-Rao norm of over-parametrize MLP.

# Simple deep linear model

Consider the following model of `Y|X=x`:

`Y|X=x ~ N(mu(x), 1)` with `mu(x) = w0 . w1 . x`

Compute likelihood landscapes for `X ~ N(0, 1)` and `y = f(x) = 2 . x`.
Plot the Fisher-Rao landscape on the (w0, w1) plane.
Compare with the empirical test likelihood or MSE on the same plane.

# A-SGD and generalization

Intuition: A-SGD should be extra beneficial when sample size is too low and model overfits (e.g. noisy labels): in this case it should be possible to use a constant large learning rate & low batch size to fit without overfitting and use. ASGD to converge to a good expected nll minimizer.

Similar analysis with large amounts of stochastic regularization (dropout, shake drop) and data-augmentation.

Devise a minimal / toy setting to highlight this effect and study theory on simplistic cases.
