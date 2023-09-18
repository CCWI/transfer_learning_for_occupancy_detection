# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Flatten, Dropout, \
                                    MaxPooling1D, Bidirectional, LSTM, Input, \
                                    Activation
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import numpy as np

class DACDBLSTM(Model):
    '''This class provides the implementation of the domain-adversarial CDBLSTM model 
       consisting of three components: feature encoder, task classifier, and domain classifier.
       Task classifier and domain classifier are parallel network branches. The model is trained 
       with labels for both, i.e. y_train where y_train[0] refers to task labels and y_train[1] to domain labels.
       When instantiating this model class, the model is already compiled and ready to train.
    '''
    
    # Training Parameters
    window_size = 30
    seed = 0
    batch_size = 128
    optimizer = optimizers.Adam()
    loss_weights = {'task_output': 1.0, 'domain_output': 1.0}
    verbose = 1
    
    # Hyperparameters
    filters = [200, 50]
    kernel_size = [5, 3]
    pool_size = 2
    lstm_neurons = [50, 50, 50]
    fc_neurons = [100]
    dropout_rates = [0.5]
    
    # Hyperparameters of the Domain Classifier
    branched_fc_neurons = [50]
    branched_dropout_rates = [0.1]
    
    def __init__(self, classes=2, features=1, domains=2, domain_clf_position=2, *args, **kwargs):
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
        domain_clf_position : TYPE, optional
            Position of the domain classifier within the network. The default is 2.
                1 = after CNN
                2 = after BLSTM
        '''
        super(DACDBLSTM, self).__init__()

        ## Parameter Setting
        self.classes = classes
        self.features = features
        self.domains = domains
        self.__dict__.update((k, v) for k, v in kwargs.items())
        print([k + "=" + str(v) for k, v in kwargs.items()])
        
        ## Gradient Reversal Layer (GRL) -------------------------     
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
                
        ## Model Layers ------------------------------------------

        # Feature Encoder  (CNN+BLSTM)
        self.conv1D_layer1 = Conv1D(filters=self.filters[0], kernel_size=self.kernel_size[0], 
                                   input_shape=(self.window_size, self.features),
                                   activation='relu', name='conv1D_layer1')
        self.pooling_layer1 = MaxPooling1D(pool_size=self.pool_size, 
                                     name='pooling_layer1')
        self.conv1D_layer2 = Conv1D(filters=self.filters[1], kernel_size=self.kernel_size[1], 
                                   input_shape=(self.window_size, self.features),
                                   activation='relu', name='conv1D_layer2')
        self.pooling_layer2 = MaxPooling1D(pool_size=self.pool_size, 
                                     name='pooling_layer2')
        
        self.blstm_layer1 = Bidirectional(LSTM(self.lstm_neurons[0], return_sequences=True), 
                                     name='blstm_layer1')
        self.blstm_layer2 = Bidirectional(LSTM(self.lstm_neurons[1], return_sequences=True),
                                     name='blstm_layer2')
        self.blstm_layer3 = Bidirectional(LSTM(self.lstm_neurons[2], return_sequences=False),
                                     name='blstm_layer3')
        
        # Task Classifier
        self.dropout_layer = Dropout(self.dropout_rates[0], seed=self.seed,
                                     name='dropout_layer')
        self.dense_layer = Dense(self.fc_neurons[0], activation='relu', kernel_initializer='uniform', 
                                     name='dense_layer')

        if self.classes < 3:
            self.task_output_layer = Dense(1, activation='sigmoid', name='task_output')
        else:
            self.task_output_layer = Dense(self.classes, activation='softmax', name='task_output')

        # Domain Classifier
        self.gradient_reversal_layer = GRL(trainable=False, name='gradient_reversal_layer')
        self.branched_dropout_layer = Dropout(self.branched_dropout_rates[0], seed=self.seed,
                                        name='branched_dropout_layer') 
        self.branched_dense_layer = Dense(self.branched_fc_neurons[0], activation="relu", kernel_initializer='uniform',
                                        name='branched_dense_layer')
        self.domain_output_layer = Dense(1, activation='sigmoid', name='domain_output')
        
        ## Layer Connection ------------------------------------------
        
        # Feature Encoder (CNN+BLSTM)
        model_input = Input(shape=(self.window_size, self.features))
        
        x = self.conv1D_layer1(model_input)
        x = self.pooling_layer1(x)
        x = self.conv1D_layer2(x)
        conv_output = self.pooling_layer2(x)
        
        x = self.blstm_layer1(conv_output)
        x = self.blstm_layer2(x)
        encoder_output = self.blstm_layer3(x)
        
        # Task Classifier
        a = self.dropout_layer(encoder_output)
        a = self.dense_layer(a)
        task_output = self.task_output_layer(a)
             
        # Domain Classifier
        if domain_clf_position == 1:
            b = self.gradient_reversal_layer(conv_output)
        else:
            b = self.gradient_reversal_layer(encoder_output)
        b = self.branched_dropout_layer(b)
        b = self.branched_dense_layer(b)
        domain_output = self.domain_output_layer(b)
         
        # Complete Network
        self.inputs = model_input
        self.outputs = [task_output, domain_output]
        self.output_names = ['task_output', 'domain_output']
        
        ## Model Compilation ------------------------------------------
        prediction_loss = 'binary_crossentropy' if classes < 3 else \
            'sparse_categorical_crossentropy'
        
        self.compile(optimizer=self.optimizer,
              loss={'task_output': prediction_loss, 
                    'domain_output':'binary_crossentropy'},
              loss_weights=self.loss_weights, metrics=['accuracy'])
        # Subnet: Task Classifier
        self.task_classifier = Model(model_input, task_output)
        self.task_classifier.compile(optimizer=self.optimizer, loss=prediction_loss, metrics=['accuracy'])
        # Subnet: Domain Classifier 
        self.domain_classifier = Model(model_input, domain_output)
        self.domain_classifier.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # Subnet: Feature Encoder
        self.encoder_model = Model(model_input, encoder_output)
        
                
    def call(self, inputs):
        return self.task_classifier(inputs), self.domain_classifier(inputs)
    
    def predict_classes(self, x, prediction_threshold=0.5):
        y_proba = self.predict(x)
        y_pred = np.array([int(p > prediction_threshold) for p in y_proba[1]])
        return y_pred
    
    def predict_tasks(self, x, prediction_threshold=0.5):
        y_proba = self.predict(x)
        y_pred = np.array([int(p > prediction_threshold) for p in y_proba[0]])
        return y_pred
        