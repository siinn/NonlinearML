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
#INPUT_PATH = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/Data_EM_extended.csv'
INPUT_PATH = '../EnhancedDividend/data/Data_EM_extended.csv'

# Set features of interest
feature_x = 'DividendYield'
feature_y = 'Payout_E'
# Set limits of decision boundary
db_xlim = (0, 0.2)
db_ylim = (-1, 1.5)
db_res = 0.0005


## Set features of interest
#feature_x = 'DY_dmed'
#feature_y = 'PO_dmed'
## Set limits of decision boundary
#db_xlim = (-1.5, 4)
#db_ylim = (-3, 4)
#db_res = 0.01


## Set features of interest
#feature_x = 'DividendYield'
#feature_y = 'EG'
## Set limits of decision boundary
#db_xlim = (0, 0.2)
#db_ylim = (-0.5, 0.5)
#db_res = 0.0005


## Set features of interest
#feature_x = 'DY_dmed'
#feature_y = 'EG_dmed'
## Set limits of decision boundary
#db_xlim = (-1.5, 4)
#db_ylim = (-4, 3)
#db_res = 0.01





# Set path to save output figures
output_path = 'output/%s_%s/cla/' % (feature_x, feature_y)
tfboard_path='tf_log/%s_%s/cla/' % (feature_x, feature_y)

# Set labels for clasification
n_classes=10
class_label={x:'D'+str(x) for x in range(n_classes)}
class_order = [x for x in range(n_classes-1, -1, -1)] # High return to low return

# Set output label classes
label_reg = 'fqRet' # continuous target label
#label_cla = 'QntfqRet' # discretized target label
label_cla = 'fqRet_discrete' # discretized target label
label_fm = 'fmRet' # monthly return used for calculating cum. return

# Set data column
date_column = "smDate"

# Set security ID column
security_id = 'SecurityID'

# Set train and test period
test_begin = "2012-01-01"
test_end = "2019-05-01"

# Set cross-validation configuration
k = 3          # Must be > 1
n_epoch = 1
subsample = 0.5
purge_length = 3

# Set p-value threshold for ANOVA test p_thres = 0.05
p_thres = 0.05

# Set metric for training
cv_metric = ['f1-score', 'precision', 'recall', 'accuracy']

# Set color scheme for decision boundary plot
#cmap = matplotlib.cm.get_cmap('Spectral', 10)
cmap = matplotlib.cm.get_cmap('RdYlGn', n_classes)
db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# Set decision boundary plotting options
db_figsize= (10, 8)
db_annot_x=0.02
db_annot_y=0.98
db_nbins=50

# Set algorithms to run
run_lr = True
run_lr_rank = False
run_xgb = False
run_knn = False
run_nn = False
run_comparison = False
save_prediction = False

#-------------------------------------------------------------------------------
# Convert configuration as a dictionary
#-------------------------------------------------------------------------------
config = {
    'feature_x':feature_x, 'feature_y':feature_y,
    'output_path':output_path, 'security_id':security_id,
    'n_classes':n_classes, 'class_label':class_label, 'class_order':class_order,
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

    # Discretize target
    df = utils.discretize_variables_by_month(
        df, variables=[config['label_reg']], n_classes=config['n_classes'],
        class_names=config['class_label'], suffix="discrete",
        month=config['date_column'])

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
            "max_iter":[50],
            "tol": [1e-2],
            "n_jobs":[-1],
            #"C": [1] + np.logspace(-4, 4, 10), # C <= 1e-5 doesn't converge
            "C": [1,2,3],
            "fit_intercept": [True]}

        # Set model
        model_lr = LogisticRegression()
        model_lr_str = 'lr'

        # Run analysis on 2D decision boundary
        db_lr = classification.decision_boundary2D(
            config, df_train, df_test,
            model_lr, model_lr_str, param_grid_lr, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=True,
            plot_decision_boundary=True,
            save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5))

    #---------------------------------------------------------------------------
    # Linear regression + ranking
    #---------------------------------------------------------------------------
    if run_lr_rank:
        # Set parameters to search
        param_grid_lr = {
            #"alpha": np.logspace(0, 4, 10)} # note: C <= 1e-5 doesn't converge
            "alpha": [1,2]}

        # Set model
        model_lr_rank = linearRank.LinearRank(
            n_classes=config['n_classes'],
            class_names=config['class_order'][::-1])
        model_lr_rank_str = 'lr_rank'

        # Run analysis on 2D decision boundary
        db_lr = classification.decision_boundary2D(
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
        ## Set parameters to search
        #param_grid_xgb = {
        #    'min_child_weight': [1000], #[1000, 500], 
        #    'max_depth': [5, 10],
        #    'eta': [0.3, 0.01, 0.1, 0.5], #[0.3],
        #    'n_estimators': [50, 100], #[50, 100, 200],
        #    'objective': ['multi:softmax'],
        #    'gamma': [0], #[0, 5, 10],
        #    'lambda': [1], #np.logspace(0, 2, 3), #[1], # L2 regularization
        #    'n_jobs':[-1],
        #    'subsample': [1], # [1]
        #    'num_class': [n_classes]}

        ## Set model
        #model_xgb = XGBClassifier()
        #model_xgb_str = 'xgb_eta'

        # Set parameters to search
        param_grid_xgb = {
            'min_child_weight': [1500, 1000, 750], #[1000, 500], #[1000], 
            'max_depth': [5, 7, 10],
            'eta': [0.3], #[0.3]
            'n_estimators': [50], # [50, 100, 200],
            'objective': ['multi:softmax'],
            'gamma': [0], #[0, 5, 10],
            'lambda': [1], #np.logspace(0, 2, 3), #[1], # L2 regularization
            'n_jobs':[-1],
            'subsample': [1, 0.8],#[1, 0.8, 0.5], # [1]
            'num_class': [n_classes]}

        # Set model
        model_xgb = XGBClassifier()
        model_xgb_str = 'xgb'


        # Run analysis on 2D decision boundary
        db_xgb = classification.decision_boundary2D(
            config, df_train, df_test,
            model_xgb, model_xgb_str, param_grid_xgb, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=True,
            plot_decision_boundary=True,
            save_csv=True,
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
            read_last=False,
            cv_study=True,
            run_backtest=True,
            plot_decision_boundary=True,
            save_csv=True,
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
            for i in range(50):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=10, activation='softmax',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            param_grid = {
                'learning_rate': [1e-3, 5e-4], #np.logspace(-4,-2,6),
                'metrics': [
                    tf.keras.metrics.SparseCategoricalCrossentropy()],
                    #tf.keras.metrics.SparseCategoricalAccuracy()],
                'loss': [tf.losses.SparseCategoricalCrossentropy()],
                #'patience': [5, 20, 200], # 3,4,5
                'patience': [1, 3, 5], # 3,4,5
                'epochs': [1000],
                'validation_split': [0.2],
                'batch_size': [64],
                'model': [ensemble_model]
                }

            # Build model and evaluate
            model = tfmodel.TensorflowModel(
                model=None, params={}, log_path=tfboard_path)
            model_str = 'nn'

            # Run analysis on 2D decision boundary
            db = classification.decision_boundary2D(
                config, df_train, df_test,
                model, model_str, param_grid, best_params={},
                read_last=False,
                cv_study=True,
                run_backtest=True,
                plot_decision_boundary=True,
                save_csv=True,
                return_train_ylim=(-1,20), return_test_ylim=(-1,5))


    #---------------------------------------------------------------------------
    # Compare model results
    #---------------------------------------------------------------------------
    models = ['lr', 'lr_rank', 'knn', 'xgb']
    if run_comparison:
        model_summary = summary.model_comparison(
            models=models,
            output_path=output_path,
            label_reg=config['label_reg'],
            class_label=config['class_order'],
            date_column=config['date_column'],
            col_pred='pred_rank',
            ylim=(-0.3,1.0))

    if save_prediction:
        predictions = summary.save_prediction(
            models=models,
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









