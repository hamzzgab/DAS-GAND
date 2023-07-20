import numpy as np
from keras.utils import to_categorical


def normalise_data(X: np.array = None) -> np.array:
    """
    Normalise the pixel values from 0-255 to 0-1
    :rtype np.array
    :param X: Input Image
    :return: Normalised pixel values
    """
    return X.astype('float32') / 255.0


def expand_dimension(X: np.array = None) -> np.array:
    """
    Expand the dimension of the input
    Example:
        (60000, 28, 28) -> (60000, 28, 28, 1)

    :rtype np.array
    :param X: Input Image
    :return: Input Image with dimension expanded
    """
    return np.expand_dims(X, axis=-1)


def categorise_data(y: np.array = None, num_classes: int = 10) -> np.array:
    """
    Categorise the target classes
    Example:
        y = [5]
        num_classes = 10

        output = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    :rtype np.array
    :param y: The target values
    :param num_classes: The number of classes in the target
    :return: Categorical values of the target
    """
    return to_categorical(y, num_classes=num_classes)


def preprocess_data(X: np.array =  None, y: np.array = None,
                    val_255: bool = True, exp_dims=True) -> tuple:

    """

    :param X:
    :param y:
    :param val_255:
    :param exp_dims:
    :return:
    """

    # NORMALISE DATA
    if val_255:
        X = normalise_data(X)
    else:
        X = (X + 1) / 2.0

    # EXPAND DIMENSIONS
    if exp_dims:
        X = expand_dimension(X)

    y = to_categorical(y)

    return X, y

