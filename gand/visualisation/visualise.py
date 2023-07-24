from gand.config import MLConfig
from gand.data import data

import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

import keras
from sklearn.metrics import confusion_matrix


# ----------------------------------------------------------------
# DATA DISTRIBUTION FOR THE TRAINING AND TESTING DATA ------------
# ----------------------------------------------------------------
def data_distribution(training: np.array = None, testing: np.array = None,
                      dataset_name: str = 'mnist',
                      figsize: tuple = (10, 10), fontsize: int = 16,
                      savefig: bool = False, showfig: bool = True,
                      figname='DataDistribution'):
    """

    :param training:
    :param testing:
    :param dataset_name:
    :param figsize:
    :param fontsize:
    :param savefig:
    :param showfig:
    :param figname:
    :return:
    """

    class_names = MLConfig.CLASS_NAMES[dataset_name]
    digits = np.arange(len(class_names))

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="dark", rc=custom_params)
    plt.figure(figsize=figsize)

    train_counts = np.bincount(training.squeeze())
    plt.bar(digits, train_counts, label='Train')

    # TEXT FOR TRAINING
    for i, count in enumerate(train_counts):
        plt.text(i, count // 2, str(count), ha='center', va='bottom', color='white',
                 fontweight='normal', rotation=90, fontsize=fontsize - 3)

    if testing is not None:
        test_counts = np.bincount(testing.squeeze())
        plt.bar(digits, test_counts, label='Test', bottom=train_counts)

        # TEXT FOR TESTING
        for i, count in enumerate(train_counts):
            plt.text(i, count + 150, test_counts[i], ha='center', va='bottom', color='black',
                     fontweight='normal', fontsize=fontsize - 3, rotation=90)

    # plt.xticks(ticks=np.arange(10), labels=class_names, fontsize=fontsize - 3, rotation=45)
    plt.xticks(ticks=np.arange(10), labels=[i for i in range(10)], fontsize=fontsize - 3, rotation=0)
    plt.yticks(fontsize=fontsize - 3)

    plt.title(f'Distribution of {dataset_name.upper()} Classes', fontsize=fontsize)
    plt.ylabel('Count', fontsize=fontsize - 2)
    plt.xlabel('Class', fontsize=fontsize - 2)

    plt.legend(loc='lower right', fontsize=fontsize - 1)

    if savefig:
        ml = MLConfig()
        path = ml.dataset_visualisation(dataset_name=dataset_name) / f'{figname}.png'
        plt.savefig(path, bbox_inches='tight')

    if showfig:
        plt.show()

    plt.close()


# ----------------------------------------------------------------
# METRIC PLOT FOR ACC AND LOSS -----------------------------------
# ----------------------------------------------------------------
def metric_plot(path: Path = None, history=None,
                dataset_name=None, epochs=None,
                fontsize=20, figsize=(10, 8),
                savefig=True, show_fig=False):
    hash = {
        'legend_loc': ['upper right', 'lower right'],
        'color': ['#1f77b4', '#ff7f0e'],
        'text': {
            'loss': [0.90, 0.80],
            'accuracy': [0.15, 0.05]
        }
    }

    sns.set_theme(style='darkgrid')
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    X = np.arange(epochs)

    for i in range(len(axs)):
        key = 'loss'
        if i == 1:
            key = 'accuracy'

        sns.lineplot(x=X, y=history.history[key], ax=axs[i],
                     linewidth=2.5, label='Training', color=hash['color'][0])
        sns.lineplot(x=X, y=history.history[f'val_{key}'], ax=axs[i],
                     linewidth=2.5, linestyle='--', label='Testing', color=hash['color'][1])

        axs[i].legend(loc=hash['legend_loc'][i], fontsize=fontsize)

        axs[i].set_ylabel(f'{key.title()}', fontsize=fontsize)
        axs[i].tick_params(axis='y', labelsize=fontsize - 6)

        for spine in ['top', 'right']:
            axs[i].spines[spine].set_visible(False)

        axs[i].text(0.55, hash['text'][key][0], '{}: {:.4f}'.format(key, history.history[f"{key}"][-1]),
                    transform=axs[i].transAxes, ha='right', fontsize=fontsize - 3, color=hash['color'][0])
        axs[i].text(0.55, hash['text'][key][1], 'val_{}: {:.4f}'.format(key, history.history[f"val_{key}"][-1]),
                    transform=axs[i].transAxes, ha='right', fontsize=fontsize - 3, color=hash['color'][1])

        axs[0].set_title(f'Loss and Accuracy Curves ({dataset_name})', fontsize=fontsize)
        axs[1].set_xlabel('Epochs', fontsize=fontsize)

        num_ticks = 10  # Specify the desired number of ticks
        xticks = np.linspace(0, epochs - 1, num_ticks)
        xticklabels = ['{:d}'.format(int(tick)) for tick in xticks]

        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xticklabels, fontsize=fontsize - 6)

    plt.tight_layout()
    if savefig:
        plt.savefig(path)
    if show_fig:
        plt.show()
    else:
        plt.close()


def confusion_matrix_plot(path=None, data=None, 
                          dataset_name=None,
                          name='training',
                          model=None, figsize=(5, 5),
                          fontsize=20):

    X, y = data
    y = np.argmax(y, axis=1)

    y_prediction = model.predict(X, verbose=0)
    y_prediction = np.argmax(y_prediction, axis=1)

    result = confusion_matrix(y, y_prediction)

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Create a heatmap of the confusion matrix
    sns.heatmap(result, annot=True, fmt='d',
                cmap='Blues', cbar=False, ax=ax, annot_kws={"size": fontsize-5})

    # Set labels, title, and ticks
    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_ylabel('Actual', fontsize=fontsize)
    ax.set_title(f'Confusion Matrix for {dataset_name}', fontsize=fontsize)

    ax.xaxis.set_ticklabels(MLConfig.CLASS_NAMES[dataset_name], rotation=90)
    ax.yaxis.set_ticklabels(MLConfig.CLASS_NAMES[dataset_name], rotation=0)

    ax.tick_params(axis='both', which='major', labelsize=fontsize-5)

    plt.tight_layout()
    plt.savefig(path.joinpath(f'{name}.png'))
    plt.close()


def plot_classes(dataset=keras.datasets.mnist,
                 fontsize=20, cmap='gray_r',
                 savefig=False, n=10, figname='classes'):
    ((X_train, y_train), (_, _)), dataset_name = data.load_dataset(dataset, return_name=True)

    class_names = MLConfig.CLASS_NAMES[dataset_name]

    num_idx = [np.where(y_train == i)[0][x] for i in range(10) for x in range(n)]

    if len(X_train.shape) == 3:
        CHANNELS = 1
    else:
        CHANNELS = 3

    y_train = y_train[num_idx]
    X_train = X_train[num_idx]

    X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], CHANNELS) / 255.0

    # fig, axes = plt.subplots(10, n, figsize=(n * 2, 10 * 2))
    fig, axes = plt.subplots(10, n, figsize=(20, 20))

    for i in range(10):
        for j in range(n):
            axes[i, j].imshow(X_train[i * n + j], cmap=cmap)
            axes[i, j].axis('off')

            if j == 0:
                axes[i, j].text(-0.5, 0.5, str(i), fontsize=fontsize + 5, va='center', ha='center', transform=axes[i, j].transAxes, rotation=0)
                # axes[i, j].text(-0.5, 0.5, class_names[i].title(), fontsize=fontsize + 5, va='center', ha='center', transform=axes[i, j].transAxes, rotation=0)

    plt.suptitle(f'Classes for {dataset_name.upper()} dataset', fontsize=fontsize+20)

    plt.tight_layout()

    if savefig:
        FOLDER = MLConfig.FIGURES_PATH.joinpath(f'{dataset_name}/data')
        FOLDER.mkdir(parents=True, exist_ok=True)
        plt.savefig(FOLDER.joinpath(f'{figname}.png'), bbox_inches='tight')

    plt.show()