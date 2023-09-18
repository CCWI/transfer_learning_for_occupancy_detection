# -*- coding: utf-8 -*-

import random
import numpy as np
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
    This class provides the trial procedure for keras tuner in phase 2 of the hyperparameter tuning.
    Each possible source-target combination of the datasets ['Candanedo', 'HM1', 'HM2', 'Home', 'Stjelja']
    is evaluated and then the mean value of the cohen's Kappa metric over all evaluations is used as the trial result.
    '''

    used_datasets = ['Candanedo', 'HM1', 'HM2', 'Home', 'Stjelja']

    window_size = 30
    batch_size = 128

    def __init__(self, data_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_samples = data_samples
        self.best_evaluation = {'Cohens Kappa': -1}

    def run_trial(self, trial):

        print("Trial", trial.trial_id)
        print(datetime.now(tz=tzInfo))

        hp = trial.hyperparameters

        # Build model
        hyper_model = self.hypermodel.build(hp=hp)

        # Prepare
        evaluation = None
        numberOfEvaluations = 0

        for target_dataset_name in self.used_datasets:
            for source_dataset_name in self.used_datasets:
                if source_dataset_name == target_dataset_name:
                    continue
                for repetition in range(0, 5):
                    # pick random samples
                    t = random.randint(0, len(self.data_samples[self.window_size][target_dataset_name]) - 1)
                    s = random.randint(0, len(self.data_samples[self.window_size][source_dataset_name]) - 1)
                    print(f"target: {target_dataset_name} (sample {t}), source: {source_dataset_name} (sample {s})")
                    x_tar, y_tar, x_val, y_val = self.data_samples[self.window_size][target_dataset_name][t]
                    # use only a single day from target data for training
                    x_tar = x_tar[0:1440]
                    y_tar = y_tar[0:1440]
                    x_src, y_src, _, _ = self.data_samples[self.window_size][source_dataset_name][s]
                    # use 5 days from source data for training
                    y_tar = [y_tar, np.array([[1] for i in range(0, len(y_tar))])]
                    y_src = [y_src, np.array([[0] for i in range(0, len(y_src))])]
                    y_val = [y_val, np.array([[1] for i in range(0, len(y_val))])]
                    y_train = np.concatenate((y_src, y_tar), axis=1)
                    x_train = np.concatenate((x_src, x_tar), axis=0)
                    # make sure training data array length is dividable by batch size
                    length = len(x_train) - (len(x_train) % self.batch_size)
                    x_train = x_train[:length]
                    y_train = (y_train[0][:length], y_train[1][:length])
                    length = len(x_val) - (len(x_val) % self.batch_size)
                    x_val = x_val[:length]
                    y_val = (y_val[0][:length], y_val[1][:length])
                    print(np.shape(x_train), np.shape(y_train), np.shape(x_val), np.shape(y_val))

                    with open(self.directory + 'data_log.csv', 'a+', newline='') as csv:
                        writer(csv).writerow([trial.trial_id, hyper_model.seed, self.window_size,
                                              target_dataset_name, t, source_dataset_name, s])

                    # Train
                    model_name = 'model_{}_{}_{}_{}_{}'.format(trial.trial_id, target_dataset_name, t,
                                                               source_dataset_name, s)
                    print("training", model_name)
                    hyper_model.model.fit(x_train, {"task_output": y_train[0], "domain_output": y_train[1]},
                                          validation_data=(x_val, y_val), epochs=100,
                                          batch_size=self.batch_size, verbose=2,
                                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',
                                                                                      patience=5, min_delta=0.00001,
                                                                                      verbose=1,
                                                                                      restore_best_weights=True)])
                    # Evaluate
                    print("evaluating")
                    if evaluation == None:
                        evaluation = evaluate(hyper_model.task_classifier, x_val, y_val[0],
                                              batch_size=self.batch_size)
                        print("Cohens Kappa for Dataset:", evaluation['Cohens Kappa'])
                        numberOfEvaluations = 1
                    else:
                        e = evaluate(hyper_model.task_classifier, x_val, y_val[0], batch_size=self.batch_size)
                        print("Cohens Kappa for Dataset:", e['Cohens Kappa'])
                        for k in evaluation.keys():
                            evaluation[k] = evaluation[k] + e[k]
                        numberOfEvaluations += 1

        # Evaluate Trial
        evaluation = {k: (evaluation[k] / numberOfEvaluations) for k in evaluation.keys()}  # average

        print("Trial Results:")
        print(evaluation)
        json.dump(evaluation, open(self.directory +
                                   f"/evaluations/evaluation_{hyper_model.seed}_{trial.trial_id}.json", 'w'))

        if evaluation['Cohens Kappa'] > self.best_evaluation['Cohens Kappa']:
            print("Cohen's Kappa improved from {} to {}".format(self.best_evaluation['Cohens Kappa'],
                                                                evaluation['Cohens Kappa']))
            print("Best so far!")

        self.oracle.update_trial(trial.trial_id, evaluation)