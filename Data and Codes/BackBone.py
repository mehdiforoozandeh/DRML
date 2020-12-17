import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize, space
from functools import partial
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


class hyperparameter_space():

    def __init__(self):
        
        self.RF_space = [
            space.Integer(200, 2000, name='n_estimators'),
            space.Categorical(['gini', 'entropy'], name='criterion'),
            space.Real(0.01, 0.3, name='min_samples_split')
        ]
        self.RF_names = ['n_estimators', 'criterion', 'min_samples_split']

        self.SVM_space = [
            space.Categorical(['rbf', 'linear', 'poly'], name='kernel'),
            space.Integer(1, 100, name='C'),
            space.Categorical(['scale', 'auto'], name='gamma')
        ]
        self.SVM_names = ['kernel', 'C', 'gamma']

        self.LR_space = [
            space.Categorical(['liblinear', 'lbfgs', 'newton-cg'], name='solver'),
            space.Real(0.1, 1, name='C')
        ]
        self.LR_names = ['solver', 'C']

        self.KNN_space = [
            space.Integer(2, 7, name='n_neighbors')
        ]
        self.KNN_names = ['n_neighbors']

        self.GBclf_space = [
            space.Categorical(['deviance', 'exponential'], name='loss'),
            space.Real(0.001, 2, name='learning_rate'),
            space.Integer(50, 2000, name='n_estimators'),
            space.Categorical(['friedman_mse', 'mse', 'mae'], name='criterion'),
            space.Real(0.01, 0.3, name='min_samples_split')
        ]
        self.GBclf_names = [
            'loss', 'learning_rate', 'n_estimators', 'criterion', 'min_samples_split']

        self.XGB_space = [
            space.Integer(50, 2000, name='n_estimators'),
            space.Real(0.001, 2, name='learning_rate'),
            space.Real(0.01, 0.9, name='gamma')
        ]
        self.XGB_names = ['n_estimators', 'learning_rate', 'gamma']

        self.GNB_space = [
            space.Real(1e-9, 1e-6, name='var_smoothing')
        ]
        self.GNB_names = ['var_smoothing']

        self.LRCN_space = [
            space.Integer(2, 6, name='cnn_n_layers'),
            space.Integer(32, 512, name='cnn_n_units_per_layer'),
            space.Real(0.5, 1.5, name='cnn_layers_delta'),
            space.Categorical([3, 5, 7, 9, 11], name='cnn_filter_size'),
            space.Integer(1, 3, name='cnn_strides'),
            space.Integer(2, 4, name='cnn_pooling_kernel_size'),
            space.Categorical(['max', 'avg'], name='cnn_pooling_strat'),
            space.Real(0.0, 0.5, name='cnn_DO'),
            space.Integer(1, 2, name='lstm_n_layers'),
            space.Integer(8, 128, name='lstm_n_units_per_layer'),
            space.Real(0.5, 1.5, name='lstm_layers_delta'),
            space.Real(0.0, 0.5, name='lstm_DO'),
            space.Integer(1, 4, name='dense_n_layers'),
            space.Integer(32, 256, name='dense_n_units_per_layer'),
            space.Real(0.5, 1.5, name='dense_layers_delta'),
            space.Real(0.0, 0.5, name='dense_DO'),
            space.Real(0.001, 1, name='lr'),
            space.Categorical([Adam, SGD, RMSprop], name='optim'),
            space.Integer(30, 400, name='batch_size'),
            space.Integer(30, 100, name='epochs')
        ]
        self.LRCN_names = [
            'cnn_n_layers', 'cnn_n_units_per_layer', 'cnn_layers_delta', 'cnn_filter_size', 'cnn_strides', 
            'cnn_pooling_kernel_size', 'cnn_pooling_strat', 'cnn_DO', 'lstm_n_layers',
            'lstm_n_units_per_layer', 'lstm_layers_delta', 'lstm_DO', 'dense_n_layers', 'dense_n_units_per_layer',
            'dense_layers_delta', 'dense_DO', 'lr', 'optim', 'batch_size', 'epochs'
        ]

        self.LSTM_space = [
            space.Integer(1, 3, name='lstm_n_layers'),
            space.Integer(8, 64, name='lstm_n_units_per_layer'),
            space.Real(0.5, 1.5, name='lstm_layers_delta'),
            space.Real(0.0, 0.5, name='lstm_DO'),
            space.Integer(1, 4, name='dense_n_layers'),
            space.Integer(32, 256, name='dense_n_units_per_layer'),
            space.Real(0.5, 1.5, name='dense_layers_delta'),
            space.Real(0.0, 0.5, name='dense_DO'),
            space.Real(0.005, 1, name='lr'),
            space.Categorical([Adam, SGD, RMSprop], name='optim'),
            space.Integer(30, 400, name='batch_size'),
            space.Integer(30, 100, name='epochs')
        ]
        self.LSTM_names = [
            'lstm_n_layers','lstm_n_units_per_layer', 'lstm_layers_delta', 'lstm_DO', 
            'dense_n_layers', 'dense_n_units_per_layer',
            'dense_layers_delta', 'dense_DO', 'lr', 'optim', 'batch_size', 'epochs'
        ]

        self.CNN_space = [
            space.Integer(2, 7, name='cnn_n_layers'),
            space.Integer(32, 512, name='cnn_n_units_per_layer'),
            space.Real(0.5, 1.5, name='cnn_layers_delta'),
            space.Categorical([3, 5, 7, 9, 11], name='cnn_filter_size'),
            space.Integer(1, 3, name='cnn_strides'),
            space.Integer(2, 4, name='cnn_pooling_kernel_size'),
            space.Categorical(['max', 'avg'], name='cnn_pooling_strat'),
            space.Real(0.0, 0.5, name='cnn_DO'),
            space.Integer(1, 4, name='dense_n_layers'),
            space.Integer(32, 256, name='dense_n_units_per_layer'),
            space.Real(0.5, 1.5, name='dense_layers_delta'),
            space.Real(0.0, 0.5, name='dense_DO'),
            space.Real(0.005, 1, name='lr'),
            space.Categorical([Adam, SGD, RMSprop], name='optim'),
            space.Integer(30, 400, name='batch_size'),
            space.Integer(30, 100, name='epochs')
        ]
        self.CNN_names = [
            'cnn_n_layers', 'cnn_n_units_per_layer', 'cnn_layers_delta', 'cnn_filter_size', 'cnn_strides', 
            'cnn_pooling_kernel_size', 'cnn_pooling_strat', 'cnn_DO', 
            'dense_n_layers', 'dense_n_units_per_layer', 'dense_layers_delta',
            'dense_DO', 'lr', 'optim', 'batch_size', 'epochs'
        ]

        self.Dense_space = [
            space.Integer(1, 4, name='dense_n_layers'),
            space.Integer(32, 1024, name='dense_n_units_per_layer'),
            space.Real(0.5, 2, name='dense_layers_delta'),
            space.Real(0.0, 0.5, name='dense_DO'),
            space.Real(0.005, 1, name='lr'),
            space.Categorical([Adam, SGD, RMSprop], name='optim'),
            space.Integer(32, 1024, name='batch_size'),
            space.Integer(30, 100, name='epochs')
        ]
        self.Dense_names = [
            'dense_n_layers', 'dense_n_units_per_layer', 'dense_layers_delta',
            'dense_DO', 'lr', 'optim', 'batch_size', 'epochs'
        ]

def load_gene_ds(gene_file, label_file, scale_strategy=None):  
    # options for scale_strategy:  None, minmax, gene_length
    gene_ds = pd.read_csv(gene_file)
    gene_ds = gene_ds.drop(['Unnamed: 0'], axis=1)

    label_ds = pd.read_csv(label_file)
    label_ds = label_ds.drop(['Unnamed: 0'], axis=1)

    if scale_strategy == 'minmax':
        gene_map = gene_ds.columns
        scaler = MinMaxScaler()
        gene_ds =  scaler.fit_transform(gene_ds)
        gene_ds = pd.DataFrame(gene_ds, columns=gene_map)
    
    elif scale_strategy == 'gene_length':
        gene_map = gene_ds.columns

        second_mat = []
        for i in range(len(gene_map)):
            temp_array = gene_map[i].split('_')
            len_i = int(temp_array[2]) - int(temp_array[1])
            second_mat.append([len_i for _ in range(gene_ds.shape[0])])

        second_mat = np.array(second_mat).transpose()
        gene_ds = gene_ds / second_mat

    return gene_ds, label_ds


if __name__ == '__main__':
    x, y = load_gene_ds('gene_based_DS.csv', 'Drug_labels.csv', scaled=True, scale_mode='gene_length')
    print(x)

    x, y = load_gene_ds('gene_based_DS.csv', 'Drug_labels.csv', scaled=True, scale_mode='minmax')
    print(x)