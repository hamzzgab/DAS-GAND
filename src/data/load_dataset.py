import numpy as np
from keras.utils import to_categorical


class Main:
    def __init__(self, dataset, model=None, name=None, class_names=None):
        # DATA
        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset.load_data()

        # MODEL ARCHITECTURE
        self.model = model

        # MODEL CONFIGURATION
        self.name = name
        self.class_names = class_names

        if self.class_names is None:
            self.class_names = np.unique(self.y_train)

        self.batch_size = None
        self.epochs = None
        self.history = None


def preprocess_image(image):
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255.0
    return image


class DataPreprocessor(Main):
    def __init__(self, dataset, model=None, name=None, class_names=None):
        super().__init__(dataset=dataset,
                         model=model,
                         name=name,
                         class_names=class_names)

    @staticmethod
    def get_num_channels(self, X_train):
        CHANNELS = 1
        CHANNELS = X_train.shape[3] if len(X_train.shape) == 4 else CHANNELS
        return CHANNELS

    def reshape_data(self, X_train, X_test):
        CHANNELS = self.get_num_channels(self, X_train)
        X_train = X_train.reshape((-1, X_train.shape[1], X_train.shape[2], CHANNELS))
        X_test = X_test.reshape((-1, X_test.shape[1], X_test.shape[2], CHANNELS))
        return X_train, X_test

    def normalise_data(self, X_train, X_test):
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        return X_train, X_test

    def encode_target_data(self, y_train, y_test):
        NUM_CLASSES = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
        return y_train, y_test

    def preprocess_data(self, X_train, y_train, X_test, y_test):
        # 1. RESHAPE THE DATA for BATCHES (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
        X_train, X_test = self.reshape_data(X_train, X_test)

        # 2. NORMALIZE THE DATA TO BE FLOAT AND DIVIDE IT BY 255
        X_train, X_test = self.normalise_data(X_train, X_test)

        # 2. ENCODE THE TARGET DATA CATEGORICALLY
        y_train, y_test = self.encode_target_data(y_train, y_test)

        return (X_train, y_train), (X_test, y_test)
