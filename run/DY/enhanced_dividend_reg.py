#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
import warnings
from xgboost.sklearn import XGBRegressor

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.preprocessing as prep
import NonlinearML.lib.stats as stats
import NonlinearML.lib.summary as summary
import NonlinearML.lib.utils as utils

import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.cross_validation as plot_cv

import NonlinearML.tf.model as tfmodel
import NonlinearML.tf.losses as tfloss
import NonlinearML.tf.metrics as tfmetric
import NonlinearML.xgb.objective as xgb_obj
import NonlinearML.xgb.metric as xgb_metric
import NonlinearML.interpret.regression as regression

# Supress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None 

#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/raw/Data_EM_extended.r3.csv'

# Set features of interest
feature_x = 'DY_dmed'
feature_y = 'PO_dmed'

# Set limits of decision boundary
db_xlim = (-1.5, 3)
db_ylim = (-3, 3)
db_res = 0.01

## Set features of interest
#feature_x = 'DividendYield'
#feature_y = 'Payout_E'
## Set limits of decision boundary
#db_xlim = (0, 0.2)
#db_ylim = (-1, 1.5)
#db_res = 0.0005

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

# Set number of bins for ranking
rank_n_bins=10
rank_label={x:'D'+str(x) for x in range(rank_n_bins)}
rank_order=[0,1,2,3,4,5,6,7,8,9] # low to high return
rank_top = 9
rank_bottom = 0

# Set output label classes
label_reg = 'fqRet' # continuous target label
#label_cla = 'QntfqRet' # discretized target label
label_cla = 'fqRet_discrete' # discretized target label
label_fm = 'fmRet' # monthly return used for calculating cum. return

# Winsorize label, followed by standardization with median in each month
standardize_label = True
winsorize_lower = 0.01
winsorize_upper = 0.99

# Set path to save output figures
output_path = 'output/DY/%s_%s/std_%s/reg_rank%s/' % (feature_x, feature_y, standardize_label, rank_n_bins)
tfboard_path='tf_log/%s_%s/std_%s/reg_rank%s/' % (feature_x, feature_y, standardize_label, rank_n_bins)


# Set data column
date_column = "smDate"

# Set security ID column
security_id = 'SecurityID'

# Set train and test period
train_begin = None
train_end = None
test_begin = "2019-06-01"
test_end = "2019-12-01"

# Set path to save model
model_tag = 'v1'
model_path = 'model/DY/%s_%s/%s/' % (feature_x, feature_y, model_tag)

# Cross validation configuration
train_from_future = True
force_val_length = False

# Prediction configuration
expand_training_window = False

# Set cross-validation configuration
k = 2    # Must be > 1
n_epoch = 1
subsample = 0.3
purge_length = 3

# Set p-value threshold for ANOVA test p_thres = 0.05
p_thres = 0.20

# Set metric for training
cv_metric = ['Top-Bottom-std', 'Top-Bottom', 'r2', 'mape']

# Set color scheme for decision boundary plot
cmap = matplotlib.cm.get_cmap('RdYlGn')
db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

cmap_scatter = matplotlib.cm.get_cmap('RdYlGn', rank_n_bins)
db_colors_scatter = [matplotlib.colors.rgb2hex(cmap_scatter(i)) 
    for i in range(cmap_scatter.N)]


# Set decision boundary plotting options
db_figsize= (10, 8)
db_annot_x=0.02
db_annot_y=0.98
db_nbins=50
db_vmin=-0.30
db_vmax=0.30

# Set residual plot 
residual_n_bins = 100

# Set algorithms to run
run_lr = False
run_xgb = True
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
    'rank_n_bins':rank_n_bins, 'rank_label':rank_label,
    'rank_top':rank_top, 'rank_bottom':rank_bottom, 'rank_order':rank_order,
    'label_reg':label_reg, 'label_cla':label_cla, 'label_fm':label_fm,
    'date_column':date_column, 'test_begin':test_begin, 'test_end':test_end,
    'k':k, 'n_epoch':n_epoch, 'subsample':subsample,
    'purge_length':purge_length, 'train_from_future':train_from_future,
    'force_val_length':force_val_length,
    'expand_training_window': expand_training_window,
    'db_xlim':db_xlim, 'db_ylim':db_ylim, 'db_res' :db_res, 'db_nbins':db_nbins,
    'db_vmin':db_vmin, 'db_vmax':db_vmax,
    'db_figsize':db_figsize, 'db_annot_x':db_annot_x, 'db_annot_y':db_annot_y,
    'p_thres':p_thres, 'cv_metric':cv_metric, 'db_colors':db_colors,
    'db_colors_scatter':db_colors_scatter, 'residual_n_bins':residual_n_bins}


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
        df, variables=[config['label_reg']], n_classes=config['rank_n_bins'],
        class_names=config['rank_label'], suffix="discrete",
        month=config['date_column'])

    if standardize_label:
        df[config['label_reg']] = prep.standardize_by_group(
            df, target=config['label_reg'], 
            groupby=config['date_column'],
            aggregate='median', wl=winsorize_lower, wu=winsorize_upper)

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end,
        train_begin, train_end, train_from_future=train_from_future)

    #---------------------------------------------------------------------------
    # Logistic regression
    #---------------------------------------------------------------------------
    if run_lr:
        # Set parameters to search
        param_grid_lr = {
            "alpha": [1] + np.logspace(-2, 3, 10), # C <= 1e-5 doesn't converge
            "fit_intercept": [True]}

        # Set model
        model_lr = Ridge()
        model_lr_str = 'lr'

        # Run analysis on 2D decision boundary
        rs_lr = regression.regression_surface2D(
            config, df_train, df_test,
            model_lr, model_lr_str, param_grid_lr, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=True,
            model_evaluation=True,
            plot_decision_boundary=True,
            plot_residual=True,
            save_csv=True,
            save_col=[config['security_id']])

        # Dump model as pickle
        utils.create_folder(model_path+"lr.pickle")
        with open(model_path+"lr.pickle", "wb") as f:
            pickle.dump(rs_lr['model'], f)

    #---------------------------------------------------------------------------
    # Xgboost
    #---------------------------------------------------------------------------
    if run_xgb:
        # Set parameters to search
        param_grid_xgb = {

            #'min_child_weight': [2000, 1500, 1000, 500], 
            #'min_child_weight': [1500, 100], 
            'min_child_weight': [1500], 

            #'max_depth': [3, 4],
            'max_depth': [3],

            'eta': [0.3], #[0.3]
            #'eta': [0.3, 0.6, 0.11, 0.01], #[0.3]

            #'gamma': [20, 10, 5, 0],
            #'gamma': [10, 5, 0],
            #'gamma': [5, 0],
            'gamma': [5],

            'lambda': [1],
            #'lambda': np.logspace(0, 2, 3), 

            #'n_estimators': [10, 50, 75, 100],
            'n_estimators': [50],

            'n_jobs':[-1],

            'objective':[
                #xgb_obj.log_square_error,
                'reg:squarederror'
                ],

            'subsample': [1],
            #'subsample': [0.5, 0.8, 1],

            }
        # Set model
        model_xgb = XGBRegressor()
        model_xgb_str = 'xgb/best_fixed_window'

        # Run analysis on 2D decision boundary
        rs_xgb = regression.regression_surface2D(
            config, df_train, df_test,
            model_xgb, model_xgb_str, param_grid_xgb, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=True,
            model_evaluation=True,
            plot_decision_boundary=True,
            plot_residual=True,
            save_csv=True,
            save_col=[config['security_id']],
            verbose=True)

        # Dump model as pickle
        utils.create_folder(model_path+"xgb.pickle")
        with open(model_path+"xgb.pickle", "wb") as f:
            pickle.dump(rs_xgb['model'], f)
        rs_xgb['model'].save_model(model_path+"xgb.%s" % model_tag)



    #---------------------------------------------------------------------------
    # kNN
    #---------------------------------------------------------------------------
    if run_knn:
        # Set parameters to search
        param_grid_knn = {
            'n_neighbors':
                sorted([int(x) for x in np.logspace(2, 3, 10)], reverse=True)}

        # Set model
        model_knn = KNeighborsRegressor()
        model_knn_str = 'knn'

        # Run analysis on 2D decision boundary
        rs_knn = regression.regression_surface2D(
            config, df_train, df_test,
            model_knn, model_knn_str, param_grid_knn, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=True,
            model_evaluation=True,
            plot_decision_boundary=True,
            plot_residual=True,
            save_csv=True,
            verbose=True,
            save_col=[config['security_id']])

        # Dump model as pickle
        utils.create_folder(model_path+"knn.pickle")
        with open(model_path+"knn.pickle", "wb") as f:
            pickle.dump(rs_knn['model'], f)



    #---------------------------------------------------------------------------
    # Neural net
    #---------------------------------------------------------------------------
    if run_nn:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            #-------------------------------------------------------------------
            # Define ensemble of weak leaners
            #-------------------------------------------------------------------
            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(200):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=1,
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model0 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(200):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=1,
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model1 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(200):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=1,
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model2 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(200):
                weak_learner = tf.keras.layers.Dense(units=64, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=64, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=1,
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model3 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(200):
                weak_learner = tf.keras.layers.Dense(units=64, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=64, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=64, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=1,
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model4 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(2,))
            for i in range(200):
                weak_learner = tf.keras.layers.Dense(units=128, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                weak_learner = tf.keras.layers.Dense(units=128, activation='relu',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                weak_learner = tf.keras.layers.Dropout(0.5)(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=1,
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model5 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            #-------------------------------------------------------------------

            param_grid = {
                'learning_rate': [5e-5, 1e-4, 5e-4, 1e-3], #np.logspace(-4,-2,6),
                'metrics': {
                    tfmetric.MeanLogSquaredError():'MLSE',
                    tf.keras.metrics.MeanAbsolutePercentageError():'mape'},
                'loss': {
                    tfloss.MeanLogSquaredError():'MLSE',
                    tf.keras.losses.Huber():'Huber',
                    tf.keras.losses.MeanAbsolutePercentageError():'mape'
                    },
                'patience': [1, 3, 10], # 3,4,5
                'epochs': [1000],
                'validation_split': [0.2],
                'batch_size': [1024],
                'model': {
                    ensemble_model0:'[32-32-1]*100',
                    ensemble_model1:'[32-0.5-32-0.5-1]*100',
                    ensemble_model2:'[32-0.5-32-0.5-32-0.5-1]*100',
                    ensemble_model3:'[64-0.5-64-0.5-1]*100',
                    ensemble_model4:'[64-0.5-64-0.5-64-0.5-1]*100',
                    ensemble_model5:'[128-0.5-128-0.5-1]*100',
                    },
                }

            # Build model and evaluate
            model = tfmodel.TensorflowModel(
                model=None, params={}, log_path=tfboard_path, model_type='reg')
            model_str = 'nn'

            # Run analysis on 2D decision boundary
            rs_nn = regression.regression_surface2D(
                config, df_train, df_test,
                model, model_str, param_grid, best_params={},
                read_last=False,
                cv_study=True,
                run_backtest=True,
                model_evaluation=True,
                plot_decision_boundary=True,
                plot_residual=True,
                save_csv=True,
                save_col=[config['security_id']],
                cv_hist_figsize=(40, 10)
                )

        # Dump model as pickle
        utils.create_folder(model_path+"nn.pickle")
        with open(model_path+"nn.pickle", "wb") as f:
            pickle.dump(rs_nn['model'], f)



    #---------------------------------------------------------------------------
    # Compare model results
    #---------------------------------------------------------------------------
    if run_comparison:
        model_comparison = summary.model_comparison(
            models=['lr', 'xgb', 'nn'],
            output_path=output_path,
            label_reg=config['label_reg'],
            class_label=config['class_order'],
            date_column=config['date_column'],
            col_pred='pred_rank',
            ylim=(-0.25,1))


    #---------------------------------------------------------------------------
    # Concatenate predictions to original date
    #---------------------------------------------------------------------------
    if save_prediction:
        predictions = summary.save_prediction(
            models=['lr', 'xgb', 'nn'],
            feature_x=config['feature_x'],
            feature_y=config['feature_y'],
            df_input=df,
            output_path=output_path,
            date_column=config['date_column'])

