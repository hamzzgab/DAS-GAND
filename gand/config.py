from pathlib import Path
from datetime import datetime


class MLConfig(object):
    # PATHS
    MODEL_VISUALISATION_PATH = Path.cwd().joinpath('reports/figures/models')
    FIGURES_PATH = Path.cwd().joinpath('reports/figures/')
    MODELS_PATH = Path.cwd().joinpath('reports/models/')
    LOGS = Path.cwd().joinpath('reports/logs/fit')

    CLASS_NAMES = {
        'mnist': [f'{i}' for i in range(10)],
        'fashion_mnist': ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle boot'],
        'cifar10': ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    }
    TYPE_NAMES = ['normal', 'augmented', 'gans', 'imbalanced-normal']

    BATCH_SIZE = 128

    def dataset_visualisation(self, dataset_name: str = None) -> Path:
        FOLDER = self.FIGURES_PATH.joinpath(f'{dataset_name}/data/')
        FOLDER.mkdir(parents=True, exist_ok=True)
        return FOLDER

    def figure_path(self, dataset_name: str = None, train_type: int = None,
                    epochs: int = None, name=None) -> Path:
        FOLDER = self.FIGURES_PATH.joinpath(f'{dataset_name}/{self.TYPE_NAMES[train_type]}/E_{epochs:03d}/{name}')
        FOLDER.mkdir(parents=True, exist_ok=True)
        return FOLDER

    def model_path(self, dataset_name: str = None, train_type: int = None,
                      epochs: int = None) -> Path:
        FOLDER = self.MODELS_PATH.joinpath(f'{dataset_name}/{self.TYPE_NAMES[train_type]}/E_{epochs:03d}')
        FOLDER.mkdir(parents=True, exist_ok=True)
        return FOLDER

    def models_log_path(self, dataset_name: str = None, train_type: int = None,
                        epochs: int = None, model_name: str = None) -> Path:
        now = datetime.now()
        date = now.strftime("%d-%m-%y_%T")
        return self.LOGS.joinpath(f'{dataset_name}/{self.TYPE_NAMES[train_type]}/E_{epochs:03d}/{model_name}_{date}')