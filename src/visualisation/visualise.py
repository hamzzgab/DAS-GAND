import numpy as np
import pandas as pd
from tabulate import tabulate
from numpy.random import randint

import visualkeras
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.load_dataset import Main
from src.data.load_dataset import DataPreprocessor

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


class DataEvaluator(DataPreprocessor):
    def __init__(self, dataset, model=None, name=None, class_names=None):
        super().__init__(dataset=dataset,
                         model=model,
                         name=name,
                         class_names=class_names)

    def plot_predictions(self, n=3):
        # PREPROCESS DATA
        (X_train, y_train), (X_test, y_test) = self.preprocess_data(self.X_train,
                                                                    self.y_train,
                                                                    self.X_test,
                                                                    self.y_test)

        # CREATE A 5X5 GRID OF SUBPLOTS
        fig, axes = plt.subplots(n, n, figsize=(n * 2 + 2, n * 2 + 2))

        # ADJUST THE SPACING BETWEEN SUBPLOTS
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        # GATHERING RANDOM DATA POINTS
        data = randint(0, len(X_test), n * n)

        # ITERATE OVER EACH SUBPLOT AND CUSTOMIZE
        for i, ax in enumerate(axes.flat):
            img = X_test[data[i]]
            img = np.expand_dims(img, axis=0)

            ax.imshow(img.squeeze(), cmap='gray')

            predict_value = self.model.predict(img, verbose=0)
            digit = np.argmax(predict_value)
            actual = np.argmax(y_test[data[i]])

            ax.set(xticks=[], yticks=[])

            title_color = 'green'

            if digit != actual:
                title_color = 'red'

            ax.set_title(f'Pred: {digit} | Actual: {actual}', color=title_color)

            for j in ['top', 'right', 'bottom', 'left']:
                ax.spines[j].set_visible(False)

        plt.show()

    def plot_confusion_matrix(self):
        # 1. PREPROCESS DATA
        (X_train, y_train), (X_test, y_test) = self.preprocess_data(self.X_train,
                                                                    self.y_train,
                                                                    self.X_test,
                                                                    self.y_test)

        y_test = np.argmax(y_test, axis=1)

        y_prediction = self.model.predict(X_test, verbose=0)
        y_prediction = np.argmax(y_prediction, axis=1)

        result = confusion_matrix(y_test, y_prediction)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Create a heatmap of the confusion matrix
        sns.heatmap(result, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

        # Set labels, title, and ticks
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix for {self.name}')
        ax.xaxis.set_ticklabels(self.class_names, rotation=90)
        ax.yaxis.set_ticklabels(self.class_names, rotation=0)
        plt.show()

    def plot_metric(self, figsize=(10, 6), metric='accuracy'):
        plt.figure(figsize=figsize)

        target = ['Training', 'Validation']
        color = ['blue', 'orange']
        linestyle = ['-', '--']
        val = ''

        X = [x + 1 for x in range(len(self.history.history[metric]))]

        plt.grid(True)
        sns.set(style="darkgrid")
        for i in range(2):
            y = self.history.history[val + metric]
            sns.lineplot(x=X, y=y, marker='o', markersize=6, color=color[i],
                         linewidth=2.5, linestyle=linestyle[i], label=target[i])

            value = max(y)
            if 'loss' in metric:
                value = min(y)

            max_index = y.index(value)
            plt.text(X[max_index], value, f"{value:.2f}",
                     ha='center', va='bottom')
            val += 'val_'

        # Set plot labels and title
        plt.xlabel("Epochs")
        plt.ylabel(metric.title())
        plt.title(metric.title() + ' Plot')

        plt.xticks(X, [str(x) for x in X])

        # Show the plot
        plt.show()


class DataVisualisation(Main):
    def __init__(self, dataset, model=None, name=None, class_names=None):
        super().__init__(dataset=dataset,
                         model=model,
                         name=name,
                         class_names=class_names)

    def plot(self, figsize=(7, 7), n=3):
        plt.figure(figsize=figsize)
        for i in range(n * n):
            plt.subplot(n, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.X_train[i], cmap=plt.cm.binary)
            plt.title(self.class_names[self.y_train[i]])
        plt.show()

    def get_label_df(self):
        train_df = pd.DataFrame({'training': self.y_train.squeeze()})
        test_df = pd.DataFrame({'testing': self.y_test.squeeze()})
        return train_df, test_df

    def get_label_counts(self):
        train_df, test_df = self.get_label_df()
        train_counts = train_df['training'].value_counts().sort_index()
        test_counts = test_df['testing'].value_counts().sort_index()
        return train_counts, test_counts

    def plot_label_count(self, figsize=(6, 4)):
        train_counts, test_counts = self.get_label_counts()

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Set the x-axis and y-axis labels
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')

        # Set the title
        ax.set_title(f'Distribution of Labels in {self.name} Dataset')

        # Plot the bar graph for training and testing datasets
        train_counts.plot(kind='bar', ax=ax, color='blue', alpha=0.7, label='Training')
        test_counts.plot(kind='bar', ax=ax, color='green', alpha=0.7, label='Testing')

        # Add text inside each bar
        for i, (train_val, test_val) in enumerate(zip(train_counts, test_counts)):
            ax.text(i, train_val / 2, train_val, ha='center', va='center')
            ax.text(i, train_val + test_val / 2, test_val, ha='center', va='center')

        # Add a legend
        ax.legend(loc='best')
        ax.set_xticklabels(self.class_names, rotation='vertical')

        # Display the graph
        plt.show()

    def model_visualisation(self, to_file=None, legend=True, draw_volume=True,
                            spacing=30, type_ignore=None):

        if to_file is None:
            to_file = self.name + '.png'

        visualkeras.layered_view(self.model, to_file=to_file, legend=legend,
                                 draw_volume=draw_volume, spacing=spacing,
                                 type_ignore=type_ignore)

    def __str__(self):
        data = [['Training', self.X_train.shape, self.y_train.shape],
                ['Testing', self.X_test.shape, self.y_test.shape]]
        col_names = ["Images", "Labels"]
        return tabulate(data, headers=col_names, tablefmt="grid")
