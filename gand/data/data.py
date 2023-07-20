import keras.models
import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10


def load_dataset(dataset: type(mnist) = mnist,
                 return_name: bool = False) -> tuple:
    """
    Loads the dataset provided from keras datasets and
    returns a tuple containing the data
    :rtype: tuple
    :param dataset: default=mnist dataset to be loaded
    :param return_name: default=False returns the name of the dataset
    :return: tuple of (X_train, y_train), (X_test, y_test), dataset_name
    """
    if return_name:
        return dataset.load_data(), dataset.__name__.split(".")[-1]
    return dataset.load_data()


def load_real_samples(dataset: type(mnist) = mnist):
    """

    :param dataset:
    :return:
    """
    (X_train, y), (_, _) = dataset.load_data()

    X = np.expand_dims(X_train, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5

    return X, y


def generate_real_samples(dataset: tuple = None, 
                          n_samples: int = None):
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [X, labels], y


def generate_fake_samples(g_model: type(keras.models.Sequential) = None,
                          latent_dim: int = None, n_samples: int = None):
    """

    :param g_model:
    :param latent_dim:
    :param n_samples:
    :return:
    """

    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = g_model.predict([z_input, labels_input])

    y = np.zeros((n_samples, 1))

    return [images, labels_input], y


def generate_latent_points(latent_dim: int = 100,
                           n_samples: int = 10,
                           n_classes: int = 10,
                           seed: int = 42) -> list:
    """

    :param latent_dim:
    :param n_samples:
    :param n_classes:
    :param seed:
    :return:
    """

    np.random.seed(seed)
    X = np.random.randn(latent_dim * n_samples)
    X = X.reshape(n_samples, latent_dim)
    y = np.random.randint(0, n_classes, n_samples)

    return [X, y]


def generate_fake_data(n: int = 2000, g_model = None, seed: int = 42):
    X_input, _ = generate_latent_points(latent_dim=100, n_samples=n * 10, seed=seed)
    y_input = np.asarray([i for i in range(10) for _ in range(n)])

    X = g_model.predict([X_input, y_input], verbose=1)
    
    return X, y_input

