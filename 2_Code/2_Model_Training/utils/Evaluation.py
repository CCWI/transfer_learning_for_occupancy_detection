# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 

def plot_ROC(fpr, tpr):
        plt.figure()
        plt.plot(fpr, tpr,
            color='blue', lw=2,
            label="ROC curve (area = %0.2f)" % metrics.auc(x=fpr, y=tpr),
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot([1, 0], [1, 0], linestyle='--', lw=2, color='grey')
        plt.legend(loc="lower right")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
    
def plot_PR(recall, precision, y_test):
        plt.figure()
        plt.plot(recall, precision,
            color='blue',
            lw=2,
            label="PR curve (area = %0.2f)" % metrics.auc(x=recall, y=precision),
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        no_skill_performance = len(y_test[y_test==1]) / len(y_test)
        plt.plot([0, 1], [no_skill_performance, no_skill_performance], 
                 linestyle='--', lw=2, color='grey', 
                 label="No Skill = %0.2f" % no_skill_performance)
        plt.legend(loc="lower right")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()
        
def __make_predictions(model, x_test, y_test, prediction_threshold=0.5, batch_size=None):
        y_proba = model.predict(x_test, batch_size=batch_size)
        if (type(y_proba)==tuple) or (len(np.shape(y_proba)) == 3): # consider only task predictions
            y_proba = y_proba[0] 
        if len(np.shape(y_test)) == 3:
            y_test = y_test[0]
        y_pred = np.array([int(p > prediction_threshold) for p in y_proba])
        print(stats.describe(y_pred), "\n")
        return y_proba, y_pred, y_test
    
def evaluate(model, x_test, y_test, prediction_threshold=0.5, batch_size=None, verbose=0):
        y_proba, y_pred, y_test = __make_predictions(model, x_test, y_test, prediction_threshold, batch_size)
        ## Calculate Metrics
        metric_values = {}
        metric_values["Accuracy"] = metrics.accuracy_score(y_test, y_pred)
        metric_values["Balanced Accuracy"] =metrics.balanced_accuracy_score(y_test, y_pred)
        metric_values["F1-Score"] = metrics.f1_score(y_test, y_pred)
        metric_values["Precision"] = metrics.precision_score(y_test, y_pred)
        metric_values["Recall"] = metrics.recall_score(y_test, y_pred)
        metric_values["Matthews Correlation Coefficient"] = metrics.matthews_corrcoef(y_test, y_pred)
        metric_values["Cohens Kappa"] = metrics.cohen_kappa_score(y_test, y_pred)
        metric_values["ROC-AUC"] = metrics.roc_auc_score(y_test, y_proba)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_proba)
        metric_values["PR-AUC"] = metrics.auc(x=recall, y=precision)
        if len(np.unique(y_test)) <= 2:
            metric_values["Loss"] = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_test, y_proba).numpy()
        else:
            metric_values["Loss"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_test, y_proba).numpy()
        
        if verbose > 1:
        ## Print Metrics
            for m in metric_values.items():
                print(m[0], "=", m[1])
       
        if verbose > 0:
        ## Print Confusion Matrix
            cm = metrics.confusion_matrix(y_test, y_pred)
            metrics.ConfusionMatrixDisplay(cm).plot()
        ## Print ROC Curve
            fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
            plot_ROC(fpr, tpr)
        ## Print Precision Recall Curve
            #metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
            plot_PR(recall, precision, y_test)
            
        return metric_values
        
def plot_loss(history, 
              training_loss_name='task_output_loss', 
              val_loss_name='val_task_output_loss'):
    plt.plot(history.history[training_loss_name])
    plt.plot(history.history[val_loss_name])
    plt.title('loss during training')
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    
def plot_accuracy(history,
                  training_acc_name='task_output_accuracy', 
                  val_acc_name='val_task_output_accuracy'):
    plt.plot(history.history[training_acc_name])
    plt.plot(history.history[val_acc_name])
    plt.title('accuracy during training')
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    
def plot_training(history):
    plot_accuracy(history)
    plot_loss(history)