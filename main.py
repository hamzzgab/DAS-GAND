from pathlib import Path

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model
from visualkeras import layered_view
from keras.datasets import mnist, fashion_mnist, cifar10

from gand import data, visualise, models, MLConfig, architecture

#
# from visualkeras import layered_view
#
# from keras.datasets import mnist, fashion_mnist, cifar10
# import tensorflow as tf
#
# opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)
# epochs = 1
# train_type = 0
#
# architecture.input_shape = (28, 28, 1)
# models.train_model(dataset=mnist, train_type=train_type,
#                    arch=architecture.baseline_1,
#                    opt=opt, epochs=epochs, verbose=0)

g_model = load_model(Path.cwd() / 'reports/models/cgan/mnist/gen_model_e-100.h5')

opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)
# epochs = 100
train_type = 2

models.train_model_gan(dataset=mnist, g_model=g_model,
                       arch=architecture.baseline_1, opt=opt,
                       epochs=40, amt_per_class=100)

# architecture.input_shape = (28, 28, 1)
# models.train_model(dataset=fashion_mnist, train_type=train_type,
#                    arch=architecture.baseline_1,
#                    opt=opt, epochs=epochs, verbose=0)
#
# architecture.input_shape = (32, 32, 3)
# models.train_model(dataset=cifar10, train_type=train_type,
#                    arch=architecture.baseline_1,
#                    opt=opt, epochs=epochs, verbose=0)
