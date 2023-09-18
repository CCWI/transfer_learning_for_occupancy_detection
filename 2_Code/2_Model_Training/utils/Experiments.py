# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import random
import json
from Models.DACDBLSTM import DACDBLSTM
from .Evaluation_v3 import evaluate

class Data():
    
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
class DataDomainwise():
    
    def __init__(self, x_tar_train, y_tar_train, x_tar_val, y_tar_val, 
                 x_src_train, y_src_train, x_src_val, y_src_val,
                 x_test, y_test):
        self.x_tar_train = x_tar_train
        self.y_tar_train = y_tar_train
        self.x_tar_val = x_tar_val
        self.y_tar_val = y_tar_val
        self.x_src_train = x_src_train
        self.y_src_train = y_src_train
        self.x_src_val = x_src_val
        self.y_src_val = y_src_val
        self.x_test = x_test
        self.y_test = y_test
        
class Settings():
    '''Defines the experiment settings.
        project_path: path in the file system to store results
        model_class:  class of the model to be built and evaluated
        trials:       number of repetitions of the same experiment
        initial_seed: seed value in first trial; seeds are then incremented by one with each trial
    '''
    def __init__(self, project_path, model_class,
                 trials=10, window_size=30, epochs=100, batch_size=128, verbose=1,
                 classes=2, features=1, domains=2, initial_seed=0):
        self.project_path = project_path
        self.model_class = model_class
        self.trials = trials
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.classes = classes
        self.features = features
        self.domains = domains
        self.initial_seed = initial_seed
        
def callbacks(monitor='val_loss', mode='min', patience=5):
    '''Generates a callback for early stopping'''
    return [tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience, min_delta=0.00001,
                                             verbose=1, restore_best_weights=True)]

def save_results(project_path, save_as, results, history, seed):
    '''Saves the results of an experiment'''
    if not os.path.exists(project_path + '/training_history/'):
        os.mkdir(project_path)
        os.mkdir(project_path + '/training_history/')
     # save training history
    np.save(project_path + '/training_history/' + save_as + "_history_" + str(seed), history.history)
     # save evaluation results    
    evaluation_file_path = project_path + "/" + save_as + ".json"
    if os.path.exists(evaluation_file_path): # if there are any previous results, extend these
        with open(evaluation_file_path, "r") as f:
            previous_results = json.load(f)
            previous_results.extend(results)
            results = previous_results
            print("evaluation results are updated in", evaluation_file_path)
    for i in results:       # convert values to floats before dumping
        for j in i.items():
             i[j[0]] = float(j[1])
    with open(evaluation_file_path, "w") as f:
        json.dump(results, f)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
## Experiment Classes
        
class TrainOnce():
    '''Trains and evaluates with the given data.
       Can be used for training once in the target domain or for fine-tuning in the target domain.
       For the latter, pass a pretrained model to be fine-tuned.
    '''
    
    def __init__(self, data, settings, save_as='targetOnly', pretrained_model=None,
                        classes=2, features=1):
        self.data = data
        self.settings = settings
        self.save_as = save_as
        self.pretrained_model = pretrained_model # can be set to retrain a given model
        
    def run(self):
        for i in range(self.settings.initial_seed, self.settings.initial_seed + self.settings.trials):
            print("Seed", i)
            set_seed(i)
            if self.pretrained_model == None:
                model = self.settings.model_class(classes=self.settings.classes, features=self.settings.features,
                                                  window_size=self.settings.window_size)
            else:
                model = self.pretrained_model
            history = model.fit(self.data.x_train, self.data.y_train,
                                epochs=self.settings.epochs, batch_size=self.settings.batch_size,
                                validation_data=(self.data.x_val, self.data.y_val), 
                                verbose=self.settings.verbose,
                                callbacks=callbacks())
            results = [evaluate(model, self.data.x_test, self.data.y_test, verbose=2)]
            save_results(self.settings.project_path, self.save_as, results, history, i)        
        
class PretrainingFinetuning():
    '''Trains a model with source data and then fine-tunes it with target data.
       Requires data object of type DataDomainwise.
       freeze_first_n_layers optionally allows to set a number of layers to be frozen during fine-tuning.
         (4 = freeze CNN, 7 = freeze CNN + BLSTM)
    '''
    
    def __init__(self, data, settings, freeze_first_n_layers=0,
                 save_as=['vanilla_pretrained', 'vanilla_finetuned']):
        self.data = data # DataDomainwise
        self.settings = settings
        self.freeze_first_n_layers = freeze_first_n_layers
        self.save_as = save_as
        
    def run(self):
        for i in range(self.settings.initial_seed, self.settings.initial_seed + self.settings.trials):
            # Pretraining
            print("Seed", i, " - Pretraining")
            set_seed(i)
            model = self.settings.model_class(classes=self.settings.classes, 
                                              features=self.settings.features,
                                              window_size=self.settings.window_size) 
            history_pre = model.fit(self.data.x_src_train, self.data.y_src_train,
                                    epochs=self.settings.epochs, 
                                    batch_size=self.settings.batch_size, 
                                    validation_data=(self.data.x_src_val, self.data.y_src_val), 
                                    verbose=self.settings.verbose,
                                    callbacks=callbacks())
            results_pre = [evaluate(model, self.data.x_test, self.data.y_test, verbose=2)]
            save_results(self.settings.project_path, self.save_as[0], results_pre, history_pre, i)
            
            # Layer Freezing
            if self.freeze_first_n_layers > 0:
                for layer in model.layers[0:self.freeze_first_n_layers]:
                    layer.trainable = False
                print("trainable layers:")
                print([(l.name, l.trainable) for l in model.layers])
                model.recompile()
        
            # Fine-Tuning
            print("Seed", i, " - Fine-Tuning")
            history_fine = model.fit(self.data.x_tar_train, self.data.y_tar_train,
                                     epochs=self.settings.epochs, 
                                     batch_size=self.settings.batch_size,
                                     validation_data=(self.data.x_tar_val, self.data.y_tar_val), 
                                     verbose=self.settings.verbose,
                                     callbacks=callbacks())
            results_fine = [evaluate(model, self.data.x_test, self.data.y_test, verbose=2)]
            save_results(self.settings.project_path, self.save_as[1], results_fine, history_fine, i)
        
class DomainAdversarialLearning():
    ''' Trains and evaluates the Domain-Adversarial Model with both target and source data.
        domain_clf_position: Position of the domain classifier within the network
                1 = after CNN
                2 = after BLSTM
    '''
    
    def __init__(self, data, settings, 
                 domain_clf_position=2,
                 save_as='domain_adversarial'):
        self.data = data
        self.settings = settings
        self.domain_clf_position = domain_clf_position
        self.save_as = save_as
            
    def run(self):
        for i in range(self.settings.initial_seed, self.settings.initial_seed + self.settings.trials):
            print("Seed", i)
            set_seed(i)
            model = self.settings.model_class(classes=self.settings.classes, features=self.settings.features, 
                                             domains=self.settings.domains, window_size=self.settings.window_size, 
                                             domain_clf_position=self.domain_clf_position)
            history = model.fit(self.data.x_train,
                                {"task_output": self.data.y_train[0], 
                                 "domain_output": self.data.y_train[1]}, 
                                epochs=self.settings.epochs, batch_size=self.settings.batch_size,
                                validation_data=(self.data.x_val, self.data.y_val), 
                                verbose=self.settings.verbose,
                                callbacks=callbacks())
            results = [evaluate(model, self.data.x_test, self.data.y_test, verbose=2)]
            save_results(self.settings.project_path, self.save_as, results, history, i)
        