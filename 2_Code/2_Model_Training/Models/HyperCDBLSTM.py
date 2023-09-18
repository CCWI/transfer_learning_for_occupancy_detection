# -*- coding: utf-8 -*-

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Flatten, Dropout, \
    MaxPooling1D, Bidirectional, LSTM, Input, \
    Activation


class HyperCDBLSTM():
    """
    Class used for hyperparameter tuning of the CDBLSTM model.
    It allows the model to be configured with different architectures and hyperparameters.
    """

    # Training Parameters
    window_size = 15
    seed = 42
    optimizer = 'Adam'
    batch_size = 32
    verbose = 1

    # Hyperparameters
    filters = [10]                  # number of layers is implicitly set by the array length
    kernel_size = 3
    pool_size = 2
    lstm_cells = [200, 150, 100]    # number of layers is implicitly set by the array length
    dense_neurons = [300, 200]      # number of layers is implicitly set by the array length
    dropout_rates = [0.5, 0.3]
    stateful_blstm = False

    def __init__(self, classes=2, features=1, domains=2, *args, **kwargs):
        '''
        Parameters
        ----------
        classes : TYPE, optional
            Number of output classes. The default is 2.
            2 is a binary classification (occupancy detection).
            4, e.g., can be a classification into high, medium, low, none
              (occupancy estimation).
        features : TYPE, optional
            Number of input features, e.g., sensor modalities. The default is 1 (for CO2 only).
        domains : TYPE, optional
            Number of domains (including target domain). The default is 2.
        '''

        # Parameter Setting
        self.classes = classes
        self.features = features
        self.domains = domains
        self.__dict__.update((k, v) for k, v in kwargs.items())
        print([k + "=" + str(v) for k, v in kwargs.items()])

        # Layer Numbers
        self.number_of_convolutions = len(self.filters)
        self.number_of_blstm_layers = len(self.lstm_cells)
        self.number_of_dense_layers = len(self.dense_neurons)

        # Model Layers
        self.conv1D_layers = []
        self.pooling_layers = []
        self.blstm_layers = []
        self.dropout_layers = []
        self.dense_layers = []

        for i in range(0, self.number_of_convolutions):
            self.conv1D_layers.append(Conv1D(filters=self.filters[i], kernel_size=self.kernel_size,
                                             input_shape=(self.window_size, self.features),
                                             activation='relu', name='conv1D_layer_{}'.format(i + 1)))
            self.pooling_layers.append(MaxPooling1D(pool_size=self.pool_size, name='pooling_layer_{}'.format(i + 1)))

        for i in range(0, self.number_of_blstm_layers):
            return_sequences = True if (i < self.number_of_blstm_layers - 1) else False
            self.blstm_layers.append(Bidirectional(LSTM(self.lstm_cells[i],
                                                        stateful=self.stateful_blstm,
                                                        return_sequences=return_sequences),
                                                   name='blstm_layer_{}'.format(i + 1)))
        for i in range(0, self.number_of_dense_layers):
            if self.dropout_rates[i] > 0:
                self.dropout_layers.append(Dropout(self.dropout_rates[i], seed=self.seed,
                                                   name='dropout_layer_{}'.format(i + 1)))
            self.dense_layers.append(Dense(self.dense_neurons[i], activation='relu', kernel_initializer='uniform',
                                           name='dense_layer_{}'.format(i + 1)))
        if self.classes < 3:
            self.task_output_layer = Dense(1, activation='sigmoid', name='task_output')
        else:
            self.task_output_layer = Dense(self.classes, activation='softmax', name='task_output_layer')

        ## Model
        model_input = Input(batch_shape=(self.batch_size, self.window_size, self.features))

        # Feature Encoder (CNN+BLSTM)
        x = model_input
        for i in range(0, len(self.conv1D_layers)):
            x = self.conv1D_layers[i](x)
            x = self.pooling_layers[i](x)
        for blstm_layer in self.blstm_layers:
            x = blstm_layer(x)
        # Task Classifier
        for i in range(0, len(self.dense_layers)):
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
            x = self.dense_layers[i](x)
        model_output = self.task_output_layer(x)

        self.model = Model(model_input, model_output)