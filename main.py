import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10
from gand import data, visualise, models, MLConfig, architecture

visualise.plot_classes(mnist,
                       savefig=True,
                       fontsize=30, figname='mnist')