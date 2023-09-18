# -*- coding: utf-8 -*-

import random
import json
from csv import writer
from datetime import datetime
import pytz
tzInfo = pytz.timezone('Europe/Paris')

import tensorflow as tf
import keras_tuner as kt

import sys
sys.path.insert(0, '../')
from utils.Evaluation import evaluate, plot_training, plot_accuracy, plot_loss


class KerasTunerHyperModel(kt.Tuner):
    '''
    This class provides the trial procedure for keras tuner in phase 1 of the hyperparameter tuning.
    For each of the datasets in ['Candanedo', 'HM1', 'HM2', 'Home', 'Stjelja']
    one of the prepared 5-day samples is randomly picked for evaluation and the mean value
    of the cohen's Kappa metric over all evaluations is used as the trial result.
    '''

    used_datasets = ['Candanedo', 'HM1', 'HM2', 'Home', 'Stjelja']

    # options
    choices = {
        'window_size': [15, 30, 60],
        'batch_size': [32, 64, 128],
        'optimizer': ['SGD', 'RMSprop', 'Adam', 'Adadelta']
    }

    def __init__(self, data_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_samples = data_samples
        self.best_evaluation = {'Cohens Kappa': -1}

    def run_trial(self, trial):

        print("Trial", trial.trial_id)
        print(datetime.now(tz=tzInfo))

        hp = trial.hyperparameters

        batch_size = hp.Choice('batch_size', self.choices['batch_size'])
        optimizer = hp.Choice('optimizer', self.choices['optimizer'])
        window_size = hp.Choice('window_size', self.choices['window_size'])

        # Build model
        hyper_model = self.hypermodel.build(hp=hp)
        if hyper_model.classes < 3:
            prediction_loss = 'binary_crossentropy'
        else:
            prediction_loss = 'sparse_categorical_crossentropy'
        model = hyper_model.model
        model.compile(optimizer=optimizer, loss=prediction_loss, metrics=['accuracy', 'AUC'])

        # Prepare
        evaluation = None
        for dataset_name in self.used_datasets:
            # pick a random sample
            i = random.randint(0, len(self.data_samples[window_size][dataset_name]) - 1)
            print(f"{dataset_name} (sample {i})")
            x_train, y_train, x_val, y_val = self.data_samples[window_size][dataset_name][i]
            with open(self.directory + 'data_log.csv', 'a+', newline='') as csv:
                writer(csv).writerow([trial.trial_id, hyper_model.seed, window_size, dataset_name, i])

            # Train
            model_name = 'model_{}_{}_{}'.format(trial.trial_id, dataset_name, i)
            print("training", model_name)
            model.fit(x_train, y_train, validation_data=(x_val, y_val), shuffle=False, epochs=100,
                      batch_size=batch_size, verbose=2,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                                  patience=5, min_delta=0.00001, verbose=1,
                                                                  restore_best_weights=True),
                                 # tf.keras.callbacks.ModelCheckpoint(self.directory + 'models/' + model_name + '.hdf5',
                                 #                                   save_best_only=True, monitor='val_loss', mode='min'),
                                 MyTensorboardCallback(log_dir=self.directory + 'tensorboard/' + model_name)])
            # Evaluate
            print("evaluating")
            if evaluation == None:
                evaluation = evaluate(model, x_val, y_val, batch_size=batch_size)
                print("Cohens Kappa for Dataset:", evaluation['Cohens Kappa'])
            else:
                e = evaluate(model, x_val, y_val, batch_size=batch_size)
                print("Cohens Kappa for Dataset:", e['Cohens Kappa'])
                for k in evaluation.keys():
                    evaluation[k] = evaluation[k] + e[k]

        # Evaluate Trial
        evaluation = {k: evaluation[k] / len(self.used_datasets) for k in evaluation.keys()}  # average

        print("Trial Results:")
        print(evaluation)
        json.dump(evaluation,
                  open(self.directory + f"/evaluations/evaluation_{hyper_model.seed}_{trial.trial_id}.json", 'w'))

        if evaluation['Cohens Kappa'] > self.best_evaluation['Cohens Kappa']:
            print("Cohen's Kappa improved from {} to {}".format(self.best_evaluation['Cohens Kappa'],
                                                                evaluation['Cohens Kappa']))
            print("Best so far!")

        self.oracle.update_trial(trial.trial_id, evaluation)