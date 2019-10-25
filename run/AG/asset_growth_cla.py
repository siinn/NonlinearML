#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import tensorflow as tf
import warnings
from xgboost.sklearn import XGBClassifier

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.utils as utils
import NonlinearML.lib.stats as stats
import NonlinearML.lib.summary as summary

import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.cross_validation as plot_cv

import NonlinearML.tf.model as tfmodel
import NonlinearML.model.linearRank as linearRank
import NonlinearML.interprete.classification as classification


# Supress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None 

#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '../data/\
Data_for_AssetGrowth_Context.r5.p2.csv'

# Set features of interest
""" Available features:
'GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC',
'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum' """
feature_x = 'AG'
feature_y = 'FCFA'

# Set path to save output figures
output_path = 'output/%s_%s/cla/' % (feature_x, feature_y)
tfboard_path='tf_log/%s_%s/cla/' % (feature_x, feature_y)

# Set labels
n_classes=3
class_label={0:'T1', 1:'T2', 2:'T3'}
class_order = [0, 1, 2] # High return to low return
class_top = 0
class_bottom = 2
suffix="descrite"

# Set output label classes
label_reg = "fqTotalReturn" # continuous target label
label_cla = "_".join([label_reg, suffix]) # discretized target label
label_fm = "fmTotalReturn" # monthly return used for calculating cum. return

# Set data column
date_column = "eom"

# Set security ID column
security_id = 'SecurityID'

# Set train and test period
test_begin = "2011-01-01"
test_end = "2017-11-01"

# Set cross-validation configuration
k =  2         # Must be > 1
n_epoch = 1
subsample = 0.8
purge_length = 3

# Set p-value threshold for ANOVA test
p_thres = 0.05

# Set limits of decision boundary
db_xlim = (-3, 3)
db_ylim = (-3, 3)
db_res = 0.01
db_figsize= (10,10)
db_annot_x=0.02
db_annot_y=0.98
db_nbins=50

# Set metric for training
cv_metric = ['f1-score', 'precision', 'recall', 'accuracy']

# Set color scheme for decision boundary plot
db_colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]
#cmap = matplotlib.cm.get_cmap('RdYlGn', 3)
#db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# Set algorithms to run
run_lr = False
run_lr_rank = True
run_xgb = False
run_svm = False
run_knn = False
run_nn = False
run_comparison = False
save_prediction = False



#-------------------------------------------------------------------------------
# Convert configuration as a dictionary
#-------------------------------------------------------------------------------
config = {
    'feature_x':feature_x, 'feature_y':feature_y,
    'output_path':output_path,
    'n_classes':n_classes, 'class_label':class_label, 'class_order':class_order,
    'class_top':class_top, 'class_bottom':class_bottom,
    'suffix':suffix, 'security_id':security_id,
    'label_reg':label_reg, 'label_cla':label_cla, 'label_fm':label_fm,
    'date_column':date_column, 'test_begin':test_begin, 'test_end':test_end,
    'k':k, 'n_epoch':n_epoch, 'subsample':subsample,
    'purge_length':purge_length,
    'db_xlim':db_xlim, 'db_ylim':db_ylim, 'db_res' :db_res, 'db_nbins':db_nbins,
    'db_figsize':db_figsize, 'db_annot_x':db_annot_x, 'db_annot_y':db_annot_y,
    'p_thres':p_thres, 'cv_metric':cv_metric, 'db_colors':db_colors}


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(INPUT_PATH, index_col=None, parse_dates=[date_column])

    # Discretize target label
    df = utils.discretize_variables_by_month(
        df=df, variables=[label_reg], month=date_column, n_classes=n_classes,
       class_names=sorted(np.arange(n_classes), reverse=True),
        suffix=suffix)

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end)

    #---------------------------------------------------------------------------
    # Logistic regression
    #---------------------------------------------------------------------------
    if run_lr:
        # Set parameters to search
        param_grid_lr = {
            "penalty":['l2'],
            "multi_class":['multinomial'],
            "solver":['newton-cg'],
            #"max_iter":[100],
            "max_iter":[10],
            "tol": [1e-2],
            "n_jobs":[-1],
            "C": [1]} # note: C <= 1e-5 doesn't converge
            #"C": [1] + np.logspace(-4, 4, 11)} # note: C <= 1e-5 doesn't converge

        # Set model
        model_lr = LogisticRegression()
        model_lr_str = 'lr'

        # Run analysis on 2D decision boundary
        db_lr = classification.decision_boundary2D(
            config, df_train, df_test,
            model_lr, model_lr_str, param_grid_lr, best_params={},
            read_last=False, cv_study=True, run_backtest=True,
            plot_decision_boundary=True, save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5))

    #---------------------------------------------------------------------------
    # Linear regression + ranking
    #---------------------------------------------------------------------------
    if run_lr_rank:
        # Set parameters to search
        param_grid_lr = {
            "alpha": [1]}
            #"alpha": [1] + np.logspace(-4, 4, 11)} # note: C <= 1e-5 doesn't converge

        # Set model
        model_lr_rank = linearRank.LinearRank(
            n_classes=config['n_classes'],
            class_names=sorted(config['class_order'], reverse=True))
        model_lr_rank_str = 'lr_rank'

        # Run analysis on 2D decision boundary
        db_lr_rank = classification.decision_boundary2D(
            config, df_train, df_test,
            model_lr_rank, model_lr_rank_str, param_grid_lr, best_params={},
            read_last=False, cv_study=True, run_backtest=True,
            plot_decision_boundary=True, save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5),
            rank=True)

    #---------------------------------------------------------------------------
    # Xgboost
    #---------------------------------------------------------------------------
    if run_xgb:
        # Set parameters to search
        param_grid_xgb = {
            'gamma': [10], #np.logspace(-2, 1, 1), # Min loss reduction
            'min_child_weight': [1500, 1000, 500],
            'max_depth': [5, 7, 10],
            'learning_rate': [0.3],
            'n_estimators': [50],
            'objective': ['multi:softmax'],
            'lambda': [1], #np.logspace(0, 2, 2) # L2 regularization
            'n_jobs':[-1],
            'num_class': [3]}

        # Set model
        model_xgb = XGBClassifier()
        model_xgb_str = 'xgb'

        # Run analysis on 2D decision boundary
        db_xgb = classification.decision_boundary2D(
            config, df_train, df_test,
            model_xgb, model_xgb_str, param_grid_xgb, best_params={},
            read_last=False, cv_study=True, run_backtest=True,
            plot_decision_boundary=True, save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5))


    #---------------------------------------------------------------------------
    # SVM
    #---------------------------------------------------------------------------
    if run_svm:
        # Set parameters to search
        param_grid_svm = {
            'C': [0.001, 0.01],
            'kernel': ['poly'],
            'degree': [3],
            'gamma': ['auto'],
            'cache_size': [3000],
            }

        # Set model
        model_svm = SVC()
        model_svm_str = 'svm'

        # Run analysis on 2D decision boundary
        db_svm = classification.decision_boundary2D(
            config, df_train, df_test,
            model_svm, model_svm_str, param_grid_svm, best_params={},
            read_last=False, cv_study=True, run_backtest=True,
            plot_decision_boundary=True, save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5))

    #---------------------------------------------------------------------------
    # kNN
    #---------------------------------------------------------------------------
    if run_knn:
        # Set parameters to search
        param_grid_knn = {
            'n_neighbors':
                sorted([int(x) for x in np.logspace(2, 3, 10)], reverse=True)}

        # Set model
        model_knn = KNeighborsClassifier()
        model_knn_str = 'knn'

        # Run analysis on 2D decision boundary
        db_knn = classification.decision_boundary2D(
            config, df_train, df_test,
            model_knn, model_knn_str, param_grid_knn, best_params={},
            read_last=True, cv_study=False, run_backtest=False,
            plot_decision_boundary=True, save_csv=False,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5))


    #---------------------------------------------------------------------------
    # Neural net
    #---------------------------------------------------------------------------
    if run_nn:

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Define ensemble of weak leaners
            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(100):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=3, activation='softmax',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            param_grid = {
                'learning_rate': [5e-4], #np.logspace(-4,-2,6),
                'metrics': [
                    tf.keras.metrics.SparseCategoricalCrossentropy()],
                    #tf.keras.metrics.SparseCategoricalAccuracy()],
                #'patience': [5, 50, 100], # 3,4,5
                'patience': [5], # 3,4,5
                'epochs': [1000],
                'validation_split': [0.2],
                'batch_size': [1024],
                'model': [
                    ensemble_model
                    ]
                }

            # Build model and evaluate
            model = tfmodel.TensorflowModel(
                model=None, params={}, log_path=tfboard_path)
            model_str = 'nn'

            # Run analysis on 2D decision boundary
            db = classification.decision_boundary2D(
                config, df_train, df_test,
                model, model_str, param_grid, best_params={},
                read_last=False, cv_study=False, run_backtest=True,
                plot_decision_boundary=True, save_csv=True,
                return_train_ylim=(-1,20), return_test_ylim=(-1,5))



    #---------------------------------------------------------------------------
    # Compare model results
    #---------------------------------------------------------------------------
    if run_comparison:
        model_comparison = summary.model_comparison(
            models=['lr', 'lr_rank', 'knn', 'xgb'],
            output_path=output_path,
            label_reg=config['label_reg'],
            class_label=config['class_order'],
            date_column=config['date_column'],
            ylim=(-0.25,1))

    #---------------------------------------------------------------------------
    # Concatenate predictions to original date
    #---------------------------------------------------------------------------
    if save_prediction:
        predictions = summary.save_prediction(
            models=['lr', 'lr_rank', 'knn', 'xgb'],
            feature_x=config['feature_x'],
            feature_y=config['feature_y'],
            df_input=df,
            output_path=output_path,
            date_column=config['date_column'])

        for model in models:
            predictions['pred_train'][model]
            print(model)
            _sum = predictions['pred_train'][model]['pred'].value_counts().sum()
            top = predictions['pred_train'][model]['pred'].value_counts()[10]
            bot = predictions['pred_train'][model]['pred'].value_counts()[1]
            print(str(top)+","+str(bot))
            print(_sum)
            print('Top %% = %s' % str(top / _sum))
            print('Bottom %% = %s' % str(bot / _sum))

        




