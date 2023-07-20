import logging

import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from gand.data import data
from gand.visualisation import visualise
from gand.config import MLConfig
from gand.preprocessing import utils

import tensorflow_addons as tfa
from tqdm.keras import TqdmCallback

tf.get_logger().setLevel(logging.ERROR)
mlconfig = MLConfig()


def save_metrics(dataset_name=None, train_type=None, epochs=None,
                 model=None, eval_data=None, history=None, mlconfig=mlconfig):
    (X_train, y_train), (X_test, y_test) = eval_data

    acc_loss_path = mlconfig.figure_path(dataset_name=dataset_name, train_type=train_type,
                                         epochs=epochs, name='acc_loss_plot').joinpath(f'{model.name}.png')
    visualise.metric_plot(path=acc_loss_path, history=history,
                          dataset_name=dataset_name, epochs=epochs)

    confusion_matrix_path = mlconfig.figure_path(dataset_name=dataset_name, train_type=train_type,
                                                 epochs=epochs, name='confusion_matrix_plot').joinpath(f'{model.name}')
    confusion_matrix_path.mkdir(parents=True, exist_ok=True)

    visualise.confusion_matrix_plot(path=confusion_matrix_path, data=(X_train, y_train),
                                    dataset_name=dataset_name, name='training',
                                    model=model, figsize=(10, 10))
    visualise.confusion_matrix_plot(path=confusion_matrix_path, data=(X_test, y_test),
                                    dataset_name=dataset_name, name='testing',
                                    model=model, figsize=(10, 10))

    model_path = mlconfig.model_path(dataset_name=dataset_name, train_type=train_type,
                                     epochs=epochs).joinpath(f'{model.name}.h5')
    model.save(model_path)


def train_model_gan(dataset: type(keras.datasets.mnist) = keras.datasets.mnist,
                    g_model: type(keras.models.Sequential) = None,
                    arch: type(keras.models.Sequential) = None,
                    opt: type(tf.keras.optimizers.legacy.SGD) = None,
                    verbose: int = 0, epochs: int = 100, batch_size: int = 128,
                    amt_per_class: int = 6000) -> None:
    train_type = 2

    ((X_train, y_train), (X_test, y_test)), dataset_name = data.load_dataset(dataset, return_name=True)

    # PREPROCESSING THE DATA
    X_train = utils.normalise_data(X_train)
    X_train = utils.expand_dimension(X_train)

    X_test = utils.normalise_data(X_test)
    X_test = utils.expand_dimension(X_test)

    # CREATE RANDOM DATA FOR DATA GENERATION
    latent_dim, n_samples = 100, amt_per_class * 10
    X, y = data.generate_latent_points(latent_dim=latent_dim,
                                       n_samples=n_samples,
                                       n_classes=10)

    y = np.asarray([i for i in range(10) for j in range(amt_per_class)])
    X = g_model.predict([X, y])
    X = (X + 1) / 2.0

    # SPLIT THE DATA FOR TRAINING AND TESTING
    X_train_gan, X_test_gan, y_train_gan, y_test_gan = train_test_split(X, y, test_size=0.2,
                                                                        shuffle=True)

    # ADD GENERATED DATA FOR TO REAL DATA
    X_train_new = np.concatenate((X_train, X_train_gan), axis=0)
    y_train_new = np.concatenate((y_train, y_train_gan), axis=0)

    X_test_new = np.concatenate((X_test, X_test_gan), axis=0)
    y_test_new = np.concatenate((y_test, y_test_gan), axis=0)

    # SHUFFLE THE DATA
    X_train, y_train = shuffle(X_train_new, y_train_new)
    X_test, y_test = shuffle(X_test_new, y_test_new)

    # CATEGORISE TARGET VARIABLES
    y_train = utils.categorise_data(y_train, num_classes=10)
    y_test = utils.categorise_data(y_test, num_classes=10)

    training, testing = (X_train, y_train), (X_test, y_test)

    model, history = fit_model(arch=arch, dataset_name=dataset_name, opt=opt, mlconfig=mlconfig,
                               train_type=train_type, epochs=epochs,
                               verbose=verbose, batch_size=batch_size, fit_data=(training, testing))

    # MODEL EVALUATION
    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    loss, acc = model.evaluate(X_test, y_test, verbose=0, callbacks=[tqdm_callback])
    print("{} | loss: {:.4f} | accuracy: {:.4f}".format(model.name, loss, acc))

    save_metrics(dataset_name=dataset_name, train_type=train_type, epochs=epochs, model=model,
                 eval_data=(training, testing), history=history, mlconfig=mlconfig)


def train_model(dataset: type(keras.datasets.mnist) = keras.datasets.mnist, train_type: int = None,
                arch: type(keras.models.Sequential) = None, opt: type(tf.keras.optimizers.legacy.SGD) = None,
                datagen: type(tf.keras.preprocessing.image.ImageDataGenerator) = None, verbose: int = 0, epochs: int = 100, batch_size: int = 128) -> None:
    """

    Trains the model for image classification, no need to perform prior preprocessing/normalisation.
    Directly saves the models analysis and the model to the path specified in the configs section.

    :param dataset: keras.datasets, default=keras.datasets.mnist
        - A keras dataset for the model to be trained on

    :param train_type: int, default=None
        - Indicating the type of training that will be performed
        - 0 -> Normal
        - 1 -> Augmented
        - 2 -> GANs (do not use this function instead use models.train_model_gan)

    :param datagen:

    :param arch: keras.models.Sequential, default=None
        - Model architecture from `architecture`
        - Use by specifying `architecture.model_name`

    :param opt: tf.keras.optimizers.legacy, default=tf.keras.optimizers.legacy.SGD(learning_rate)
        - Optimizer to be used for training the dataset

    :param verbose: int, default=0

    :param epochs: int, default=100

    :param batch_size: int, default=128

    :return: None
    """

    ((X_train, y_train), (X_test, y_test)), dataset_name = data.load_dataset(dataset, return_name=True)

    # PREPROCESSING THE DATA
    X_train = utils.normalise_data(X_train)
    y_train = utils.categorise_data(y_train, num_classes=10)

    X_test = utils.normalise_data(X_test)
    y_test = utils.categorise_data(y_test, num_classes=10)

    if len(X_train.shape) == 3:
        X_train = utils.expand_dimension(X_train)
        X_test = utils.expand_dimension(X_test)

    training, testing = (X_train, y_train), (X_test, y_test)

    model, history = fit_model(arch=arch, dataset_name=dataset_name, opt=opt, mlconfig=mlconfig,
                               train_type=train_type, epochs=epochs,
                               verbose=verbose, batch_size=batch_size, fit_data=(training, testing),
                               datagen=datagen)

    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    loss, acc = model.evaluate(X_test, y_test, verbose=0, callbacks=[tqdm_callback])
    print("{} | loss: {:.4f} | accuracy: {:.4f}".format(model.name, loss, acc))

    save_metrics(dataset_name=dataset_name, train_type=train_type, epochs=epochs, model=model,
                 eval_data=(training, testing), history=history, mlconfig=mlconfig)


def fit_model(arch=None, dataset_name=None, opt=None, mlconfig=mlconfig, train_type=None, epochs=None,
              verbose=None, batch_size=None, fit_data=None, datagen=None):
    (X_train, y_train), (X_test, y_test) = fit_data
    # GET THE MODEL ARCHITECTURE
    model = arch()

    # COMPILE THE MODEL
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    log_dir = mlconfig.models_log_path(dataset_name=dataset_name, train_type=train_type,
                                       epochs=epochs, model_name=model.name)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if train_type == 0:
        history = model.fit(X_train, y_train, epochs=epochs,
                            validation_data=(X_test, y_test),
                            batch_size=batch_size,
                            callbacks=[tensorboard_callback, TqdmCallback(verbose=verbose)],
                            verbose=verbose)
    else:
        history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                            validation_data=datagen.flow(X_test, y_test, batch_size=8),
                            epochs=epochs, callbacks=[tensorboard_callback, TqdmCallback(verbose=verbose)],
                            verbose=verbose)

    return model, history
