import torch.nn as nn
import numpy as np
np.random.seed(42)
import tensorflow as tf

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense

import numpy as np

MAX_INT = np.iinfo(np.int32).max
data_format = 0


def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=(input_shape,))
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(2, activation='softmax',  name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_s(input_shape, output_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=(input_shape,))
    # intermediate = Dense(20, activation='relu',
    #             kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(2*input_shape, activation='relu', name='hl1')(x_input)
    intermediate = Dense(input_shape, activation='relu', name='hl2')(intermediate)
    intermediate = Dense(output_shape, activation='softmax',  name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_s_r(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=(input_shape,))
    # intermediate = Dense(20, activation='relu',
    #             kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(2*input_shape, activation='relu', name='hl1')(x_input)
    intermediate = Dense(input_shape, activation='relu', name='hl2')(intermediate)
    intermediate = Dense(1, name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_sr(input_shape, output_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=(input_shape,))
    # intermediate = Dense(20, activation='relu',
    #             kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(2*input_shape, activation='relu', name='hl1')(x_input)
    intermediate = Dense(input_shape, activation='relu', name='hl2')(intermediate)
    intermediate = Dense(1, name='score')(intermediate)
    return Model(x_input, intermediate)

def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 2),
            # nn.ReLU()
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        # self.layer = nn.Sequential(
        #     nn.Linear(18, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layer(x)
        return x
