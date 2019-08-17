#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import warnings

# Import custom libraries
from Asset_growth.lib.plots import *
from Asset_growth.lib.utils import *
from Asset_growth.lib.purged_k_fold import *
from Asset_growth.lib.heuristicModel import *
from Asset_growth.lib.write_log import *

# Supress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None 

#----------------------------------------------
# Set user options
#----------------------------------------------
# Set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.p2.csv'

# Set available features and labels
''' Available features: 'GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC', 'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum' '''
feature_x = 'AG'
feature_y = 'FCFA'
features = [feature_x, feature_y]

# Set path to save output figures
output_path = 'plots/%s_%s/' % (feature_x, feature_y)
TFBOARD_PATH="log/2019.08.05/"

# Set labels
label = "fqTotalReturn_tertile"     # or "fmTotalReturn_quintile"
label_reg = "fqTotalReturn"         # or "fmTotalReturn"
label_fm = "fmTotalReturn"          # Used to calculate cumulative return
date_column = "eom"

# Set number of classes
num_class = 3

# Set train and test period
test_begin = "2011-01-01"
test_end = "2017-11-01"

# Set k-fold
k = 5

# Set color scheme for decision boundary plot
colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]


#----------------------------------------------
# Create output folder
#----------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(TFBOARD_PATH):
    os.makedirs(TFBOARD_PATH)
#-----------------------------------------
# Model class
#-----------------------------------------
class Keras_Model:
    """ Tensorflow model class to perform compile, fit, and evaluation. 
    Attributes:
        model: tf.keras.Sequential model
        params: model parameters
        metrics: dictionary containing metric functions
        params: Parameters used to build model.
    """
    def __init__(self, model, metrics, params, log_path):
        """ Initialize variables."""
        print("Building model..")
        # parameter set
        self.model = model
        self.params = params
        self.metrics = metrics
        self.log_path = log_path

        # The patience parameter is the amount of epochs to check for improvement
        #self.early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.params[HP_PATIENCE])
        self.early_stop  = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=self.params[HP_PATIENCE], mode='max')
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=True, update_freq='epoch')
        #self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=True, update_freq=1000)
    
        # Set callback. Write TensorBoard logs to `./logs` directory
        self.callbacks = [self.tensorboard, self.early_stop]

    def set_params(self, params):
        """ Set model parameters. """
        self.params = params

    def compile(self):
        """ Compile model."""
        self.model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.params[HP_LEARNING_RATE]),
                      metrics=[self.metrics[self.params[HP_METRICS]]])

    def fit(self, train_x, train_y):
        """ Train model."""
        history = self.model.fit(x=train_x, y=train_y,
                            epochs=self.params[HP_EPOCHS],
                            batch_size=32,
                            validation_split=self.params[HP_VALIDATION_SPLIT],
                            callbacks=self.callbacks)
    def predict(self, x):
        """ Make prediction."""
        return self.model.predict_classes(x=x, verbose=1, batch_size=32)

    def evaluate(self, train_x, train_y, test_x, test_y):
        """ Make prediction and evaluate model with classification metrics."""
        y_pred_train = self.model.predict_classes(x=train_x, verbose=1, batch_size=32)
        y_pred_test = self.model.predict_classes(x=test_x, verbose=1, batch_size=32)
        # Calculate classification metrics
        cr_train = classification_report(train_y, y_pred_train, output_dict=True)
        cr_test = classification_report(test_y, y_pred_test, output_dict=True)
        return cr_train, cr_test


def extract_metrics(class_report, num_class):
    ''' Extract metrics from sklearn classification report.
    Args:
        class_report: Classification report from sklearn.metrics.
        num_class: Number of classes
    Return:
        Dictionary containing evaluation metrics.
    '''
    # Dictionary to hold results
    results = {}
    # List of classes = classificatin labels + macro and micro average
    list_classes = [str(float(i)) for i in range(num_class)] + ['macro avg', 'weighted avg']
    # Get metrics for each classes 
    for cls in list_classes:
        results['train_%s_precision' % cls] = class_report[cls]['precision']
        results['train_%s_recall' % cls]    = class_report[cls]['recall']
        results['train_%s_f1-score' % cls]  = class_report[cls]['f1-score']
    # Get overall accuracy
    results['accuracy'] = class_report['accuracy']
    return results


#------------------------------------------
# Main 
#------------------------------------------
if __name__ == "__main__":

    print('Running two factor classification with the following two factors:')
    print(' > feature x: %s' % feature_x)
    print(' > feature y: %s' % feature_y)

    #------------------------------------------
    # Read dataset
    #------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=[0], parse_dates=[date_column])

    # Assign tertile labels to the features
    df = discretize_variables_by_month(df=df, variables=features, month="eom",
                                       labels_tertile={feature_x:[2,1,0], feature_y:[2,1,0]}, # 0 is high
                                       labels_quintile={feature_x:[4,3,2,1,0], feature_y:[4,3,2,1,0]}) # 0 is high
    tertile_boundary =  get_tertile_boundary(df, features)

    # Split dataset into train and test dataset
    df_train, df_test = train_test_split_by_date(df, date_column, test_begin, test_end)


    # Create directory to save figures
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #------------------------------------------
    # Prepare input data for TF
    #------------------------------------------
    # Convert dataframe to numpy as input for TF
    train_x = df_train[features].values
    train_y = df_train[label].values
    test_x = df_test[features].values
    test_y = df_test[label].values


    #------------------------------------------
    # Deep learning
    #------------------------------------------

    # Define training metrics
    metrics = {'sparse_categorical_cross_ent':tf.keras.metrics.SparseCategoricalCrossentropy(),
               'sparse_categorical_accuracy':tf.keras.metrics.SparseCategoricalAccuracy()}

    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.5]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-6])) #np.logspace(-4,-2,6),
    HP_METRICS = hp.HParam('metrics', hp.Discrete(['sparse_categorical_accuracy'])) #'sparse_categorical_cross_ent'
    HP_PATIENCE = hp.HParam('patience', hp.Discrete([3,4,5]))
    #HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300]))
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete([100]))
    HP_VALIDATION_SPLIT = hp.HParam('validation_split', hp.Discrete([0.2]))

    HPARAMS = [ HP_DROPOUT,
                HP_LEARNING_RATE,
                HP_METRICS,
                HP_PATIENCE,
                HP_EPOCHS,
                HP_VALIDATION_SPLIT,
                ]

    # Define evaluation metrics
    eval_metrics = ['train_0.0_precision',
                    'train_0.0_recall', 
                    'train_0.0_f1-score', 
                    'train_1.0_precision',
                    'train_1.0_recall', 
                    'train_1.0_f1-score', 
                    'train_2.0_precision',
                    'train_2.0_recall', 
                    'train_2.0_f1-score', 
                    'accuracy']


    # Register metrics to HParams
    METRICS = [hp.Metric(metric_name, display_name=metric_name) for metric_name in eval_metrics]

    # Writa top-level experiment configuration.
    with tf.summary.create_file_writer(TFBOARD_PATH).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    # Get all possible combination of parameters
    keys, values = zip(*{x:x.domain.values for x in HPARAMS}.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Define dataframe to hold results
    grid_train_results = pd.DataFrame()
    grid_test_results = pd.DataFrame()

    # Set evaluation metric in choosing best model
    best_param_metric = 'f1-score'
    best_model = {}

    # Loop over parameter sets
    n_experiments = len(experiments)
    count = 0
    for params in experiments:
        # Add count
        print("Training model (%s/%s)" % (count+1, n_experiments))
        count = count+1

        # Set tensorboard log path
        HPARAM_PATH = TFBOARD_PATH+'%s_%s' % (str(datetime.datetime.now()).replace(' ', '_'),
                                              str([x.name+"="+str(params[x]) for x in params]).replace('\'', '').replace(' ', ''))

        # Define model
        model = tf.keras.Sequential([
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(params[HP_DROPOUT]),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(params[HP_DROPOUT]),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(num_class, activation='softmax')
                        ])

        # Build model and evaluate
        model = Keras_Model(model=model, metrics=metrics, params=params, log_path=HPARAM_PATH)
        model.compile()
        model.fit(train_x=train_x, train_y=train_y)

        # Evaluate model and store results
        cr_train, cr_test = model.evaluate(train_x, train_y, test_x, test_y)

        # Extract metrics from classification report
        train_results = extract_metrics(cr_train, num_class)
        test_results = extract_metrics(cr_test, num_class)

        # Combine metrics with parameters
        train_results = dict([(x.name, model.params[x]) for x in model.params] + list(train_results.items()))
        test_results = dict([(x.name, model.params[x]) for x in model.params] + list(test_results.items()))

        # Append grid search results
        grid_train_results = grid_train_results.append(pd.DataFrame.from_dict(train_results, orient="index").T)
        grid_test_results = grid_test_results.append(pd.DataFrame.from_dict(test_results, orient="index").T)

        # Select best parameters
        ''' Change this to validation!'''
        if len(grid_test_results) == 0:    # First result
            best_model['params'] = params
            best_model['model'] = model
        elif grid_test_results[best_param_metric].max() <= test_results[best_param_metric]: # Check if current result is the best
            best_model['params'] = params
            best_model['model'] = model

        # Write evaluation metrics for hparams
        with tf.summary.create_file_writer(HPARAM_PATH).as_default():
            print("Writing parameters to hparams") 
            print(params)
            hp.hparams(params)  
            for metric_name in eval_metrics:
                tf.summary.scalar(name=metric_name, data=train_results[metric_name], step=1)

    #------------------------------------
    # Write log
    #------------------------------------
    # Save configuration to file
    write_log(features, input_path, TFBOARD_PATH, model, grid_train_results, grid_test_results,
              test_begin, test_end)

    #------------------------------------
    # Evaluate model
    #------------------------------------
    if True:
        # Calculate cumulative return using best parameters
        df_cum_train_tf, df_cum_test_tf, model_tf = predict_and_calculate_cum_return(model=best_model['model'], 
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm, refit=False)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_tf, df_cum_test_tf, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename=output_path+"return_tf", kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_tf], return_label=[0,1,2],
                                    list_labels=["Neural net"], label_reg=label_reg, figsize=(8,6),
                                    filename=output_path+"return_tf_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})
    
        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_tf, df=df_test, features=features, h=0.01, x_label=feature_x, y_label=feature_y,
                               vlines=tertile_boundary[feature_x], hlines=tertile_boundary[feature_y], colors=colors,
                               #xlim=(-3,3), ylim=(-3,3), figsize=(8,6), ticks=[0,1,2], filename=output_path+"decision_boundary_tf")
                               xlim=None, ylim=None, figsize=(8,6), ticks=[0,1,2], filename=output_path+"decision_boundary_tf")

        # Save cumulative returns as pickle   
        pickle.dump(df_cum_test_tf, open(output_path+'df_cum_test_tf.pickle', 'wb'))


    print("Successfully completed all tasks")












#
#
#
#
#
#
#

