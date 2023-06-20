import math
import numpy as np

from keras.callbacks import LearningRateScheduler, EarlyStopping

from src.visualisation.visualise import DataVisualisation, DataEvaluator


class Model(DataVisualisation, DataEvaluator):
    def __init__(self, dataset, model=None, name=None, class_names=None):
        super().__init__(dataset=dataset,
                         model=model,
                         name=name,
                         class_names=class_names)

    @staticmethod
    def step_decay(epoch, lr):
        if epoch % 10 == 0:
            lr *= 0.1
            return lr
        return lr

    @staticmethod
    def cosine_annealing(epoch, initial_lr, total_epochs):
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / total_epochs))
        return lr

    def train_model(self, input_shape=(28, 28, 1),
                    learning_rate=0.001,
                    batch_size=64,
                    epochs=20):

        # PREPROCESS DATA
        (X_train, y_train), (X_test, y_test) = self.preprocess_data(self.X_train,
                                                                    self.y_train,
                                                                    self.X_test,
                                                                    self.y_test)

        self.batch_size = batch_size
        self.epochs = epochs

        reduce_lr = LearningRateScheduler(lambda epoch: self.cosine_annealing(epoch,
                                                                              learning_rate,
                                                                              epochs))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        self.history = self.model.fit(X_train, y_train, batch_size,
                                      epochs=self.epochs,
                                      validation_data=(X_test, y_test),
                                      verbose=1,
                                      steps_per_epoch=len(self.X_train) / batch_size,
                                      callbacks=[reduce_lr, early_stopping])

    def save_model(self, name=None):
        if name is None:
            name = self.name + '_e' + str(self.epochs) + '_bS' + str(self.batch_size) + '.h5'
        self.model.save(name)

    def predict(self, image):
        image = preprocess_image(image)
        y_prediction = self.model.predict(image, verbose=0)
        y_prediction = np.argmax(y_prediction, axis=1)
        return y_prediction.squeeze()

    def __str__(self):
        return super().__str__()