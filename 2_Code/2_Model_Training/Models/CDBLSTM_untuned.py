# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, \
    MaxPooling1D, Bidirectional, LSTM, Input
from tensorflow.keras import optimizers


class CDBLSTM(Sequential):
    '''
    This class provides the implementation of the untuned CDBLSTM model.
    Architecture and predefined hyperparameters are taken from the following publication:
        Zhenghua Chen, Rui Zhao, Qingchang Zhu, Mustafa K. Masood, Yeng Chai
        Soh, and Kezhi Mao. 2017. Building Occupancy Estimation with Environmental
        Sensors via CDBLSTM. IEEE Transactions on Industrial Electronics 64, 12 (2017),
        9549–9559.
    '''

    # Hyperparameters
    filters = 10
    kernel_size = 3
    pool_size = 2
    lstm_neurons = [200, 150, 100]
    fc_neurons = [300, 200]
    dropout_rates = [0.5, 0.3]

    def __init__(self, classes=2, features=1, window_size=15, batch_size=32, seed=0):
        '''
        Parameters
        ----------
        classes : TYPE, optional
            Number of output classes. The default is 2.
            2 is a binary classification (occupancy detection).
            4, e.g., can be a classification into high, medium, low, none
              (occupancy estimation).
        features : TYPE, optional
            Number of input features. The default is 1 for CO2 only.
        window_size : TYPE, optional
            Number of timesteps (in minutes) contained in one input sequence.
        batch_size : TYPE, optional
            Number of sequences contained in each training batch.
        seed : TYPE, optional
            Seed value used to initialize randomization of dropout.
        '''
        super(CDBLSTM, self).__init__()

        self.classes = classes
        self.features = features
        self.window_size = window_size
        self.batch_size = batch_size
        self.seed = seed

        ## Model Architecture

        # Convolutional Network
        self.add(Input(shape=(self.window_size, self.features), batch_size=self.batch_size, name='input'))
        self.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                        activation='relu', dtype='float32'))
        self.add(MaxPooling1D(pool_size=self.pool_size))

        # LSTM
        self.add(Bidirectional(LSTM(self.lstm_neurons[0], return_sequences=True, dtype='float32')))
        self.add(Bidirectional(LSTM(self.lstm_neurons[1], return_sequences=True, dtype='float32')))
        self.add(Bidirectional(LSTM(self.lstm_neurons[2], return_sequences=False, dtype='float32')))

        # Fully Connected Layers
        self.add(Dropout(self.dropout_rates[0], seed=self.seed, dtype='float32'))
        self.add(Dense(self.fc_neurons[0], activation='relu', kernel_initializer='uniform', dtype='float32'))
        self.add(Dropout(self.dropout_rates[1], seed=self.seed, dtype='float32'))
        self.add(Dense(self.fc_neurons[1], activation='relu', kernel_initializer='uniform', dtype='float32'))

        # Softmax Output Layer
        if self.classes < 3:
            self.add(Dense(1, activation='sigmoid', dtype='float32'))
        else:
            self.add(Dense(self.classes, activation='softmax', dtype='float32'))

        # Optimizer
        if self.classes < 3:
            self.compile(loss='binary_crossentropy',
                         optimizer=optimizers.RMSprop(), metrics=['Accuracy'])
        else:
            self.compile(loss='sparse_categorical_crossentropy',
                         optimizer=optimizers.RMSprop(), metrics=['Accuracy'])