# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Flatten, Dropout, \
    MaxPooling1D, Bidirectional, LSTM, Input, \
    Activation


class HyperDACDBLSTM():
    """
    Class used for hyperparameter tuning of the domain classifier in the Domain-Adversarial CDBLSTM model.
    It allows the domain classifier to be configured with different architectures and hyperparameters.
    architecture and hyperparameters of the task classifier are predefined according to the results
    of tuning phase 1.
    """

    # Training Parameters
    window_size = 30
    seed = 42
    batch_size = 128
    optimizer = 'Adam'
    verbose = 1
    loss_weights = {'task_output': 1.0, 'domain_output': 1.0}

    # Hyperparameters
    filters = [200, 50]
    kernel_size = [5, 3]
    pool_size = 2
    lstm_cells = [50, 50, 50]
    dense_neurons = [100]
    dropout_rates = [0.5]
    stateful_blstm = True

    # Hyperparameters of the Domain Classifier
    domain_clf_neurons = [100]  # number of layers is implicitly set by the array length
    domain_clf_dropout_rates = [0.5]

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
            Number of features, e.g., sensor modalities. The default is 1.
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
        self.number_of_branched_layers = len(self.domain_clf_neurons)

        # Model Layers
        self.conv1D_layers = []
        self.pooling_layers = []
        self.blstm_layers = []
        self.dropout_layers = []
        self.dense_layers = []
        self.branched_dense_layers = []
        self.branched_dropout_layers = []

        # Gradient Reversal Layer (GRL)
        class GRL(Layer):
            @tf.autograph.experimental.do_not_convert
            def __init__(self, *args, **kwargs):
                super(GRL, self).__init__(*args, **kwargs)

            @tf.custom_gradient
            def _reverse_gradient(self, x):
                y = tf.identity(x)

                def _flip_grad(grad):
                    return tf.negative(grad)

                return y, _flip_grad

            def call(self, x):
                return self._reverse_gradient(x)

        # ----------------------------------------------    Model Layers

        # Feature Extractor  (CNN+BLSTM)
        for i in range(0, self.number_of_convolutions):
            self.conv1D_layers.append(Conv1D(filters=self.filters[i], kernel_size=self.kernel_size[i],
                                             input_shape=(self.window_size, self.features),
                                             activation='relu', name='conv1D_layer_{}'.format(i + 1)))
            self.pooling_layers.append(MaxPooling1D(pool_size=self.pool_size,
                                                    name='pooling_layer_{}'.format(i + 1)))

        for i in range(0, self.number_of_blstm_layers):
            return_sequences = True if (i < self.number_of_blstm_layers - 1) else False
            self.blstm_layers.append(Bidirectional(LSTM(self.lstm_cells[i],
                                                        stateful=self.stateful_blstm,
                                                        return_sequences=return_sequences),
                                                   name='blstm_layer_{}'.format(i + 1)))

        # Task Classifier
        for i in range(0, self.number_of_dense_layers):
            if self.dropout_rates[i] > 0:
                self.dropout_layers.append(Dropout(self.dropout_rates[i], seed=self.seed,
                                                   name='dropout_layer_{}'.format(i + 1)))
            self.dense_layers.append(Dense(self.dense_neurons[i], activation='relu', kernel_initializer='uniform',
                                           name='dense_layer_{}'.format(i + 1)))
        if self.classes < 3:
            self.task_output_layer = Dense(1, activation='sigmoid', name='task_output')
        else:
            self.task_output_layer = Dense(self.classes, activation='softmax', name='task_output')

        # Domain Classifier
        self.gradient_reversal_layer = GRL(trainable=False, name='gradient_reversal_layer')

        for i in range(0, self.number_of_branched_layers):
            if self.domain_clf_dropout_rates[i] > 0:
                self.branched_dropout_layers.append(Dropout(self.domain_clf_dropout_rates[i], seed=self.seed,
                                                            name='branched_dropout_layer_{}'.format(i + 1)))
            self.branched_dense_layers.append(Dense(self.domain_clf_neurons[i], activation='relu',
                                                    kernel_initializer='uniform',
                                                    name='branched_dense_layer_{}'.format(i + 1)))

        self.domain_output_layer = Dense(1, activation='sigmoid', name='domain_output')

        # ----------------------------------------------    Connect Layers

        model_input = Input(batch_shape=(self.batch_size, self.window_size, self.features))

        # Feature Encoder (CNN+BLSTM)
        x = model_input
        for i in range(0, len(self.conv1D_layers)):
            x = self.conv1D_layers[i](x)
            x = self.pooling_layers[i](x)
        for blstm_layer in self.blstm_layers:
            x = blstm_layer(x)
        encoder_output = x

        # Task Classifier
        a = encoder_output
        for i in range(0, len(self.dense_layers)):
            if i < len(self.dropout_layers):
                a = self.dropout_layers[i](a)
            a = self.dense_layers[i](a)
        task_output = self.task_output_layer(a)

        # Domain Classifier
        b = self.gradient_reversal_layer(encoder_output)
        for i in range(0, len(self.branched_dense_layers)):
            if i < len(self.branched_dropout_layers):
                b = self.branched_dropout_layers[i](b)
            b = self.branched_dense_layers[i](b)
        domain_output = self.domain_output_layer(b)

        ## Model Compilation
        prediction_loss = 'binary_crossentropy' if classes < 3 else \
            'sparse_categorical_crossentropy'

        # Complete Network
        self.model = Model(model_input, [task_output, domain_output])
        self.model.output_names = ['task_output', 'domain_output']
        self.model.compile(optimizer=self.optimizer,
                           loss={'task_output': prediction_loss,
                                 'domain_output': 'binary_crossentropy'},
                           loss_weights=self.loss_weights,
                           metrics=['accuracy', 'AUC'])
        # Subnet: Task Classifier
        self.task_classifier = Model(model_input, task_output)
        self.task_classifier.compile(optimizer=self.optimizer,
                                     loss=prediction_loss,
                                     metrics=['accuracy', 'AUC'])
        # Subnet: Domain Classifier
        self.domain_classifier = Model(model_input, domain_output)
        self.domain_classifier.compile(optimizer=self.optimizer,
                                       loss='binary_crossentropy',
                                       metrics=['accuracy', 'AUC'])
        # Subnet: Feature Encoder
        self.encoder_model = Model(model_input, encoder_output)
