#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import matplotlib
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import warnings
from xgboost.sklearn import XGBClassifier

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.preprocessing as prep
import NonlinearML.lib.utils as utils
import NonlinearML.lib.stats as stats
import NonlinearML.lib.summary as summary

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
import NonlinearML.interpret.classification as classification

# Supress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None 

#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '../data/ASA/csv/ASA_G2_data.r5.p1.csv'

# Set input path for beta correlation data
beta_corr_path = '/mnt/mainblob/nonlinearML/data/ASA/csv/20200427_CORR.xls'
beta_corr_date = 'smDate'
beta_corr = ['PM6M', 'PM12M']
dt = 3      # Time step to calculate difference

# Set features of interest
feature_x = 'PM12M'
feature_y = 'PM12M_%sMChange' % dt

# Configuration for PM reversal flag
PM = 'PM_Exp'
DIFF = 'Diff_Exp'
PM_N_CLASS = 5
FQ_REL_RETURN = 'fqRelRet'
pm_date = 'smDate'

features = [DIFF, PM]
return_group = "%s-%s" %(DIFF, PM)
feats_disc = ["%s_n%s" % (feat, PM_N_CLASS) for feat in features]
moving_average = 6 # Moving average applied to the mean return of PM/DIFF quintiles
thres = 0.05    # Threshold in creating PM reversal flag

# Set limits of decision boundary
db_xlim = (-1, 1)
db_ylim = (-1, 1)
db_res = 0.01

# Set decision boundary plotting options
db_figsize= (10, 8)
db_annot_x=0.02
db_annot_y=0.98
db_nbins=50
db_vmin=0
db_vmax=1

# Set residual plot 
residual_n_bins = 100

# Set path to save output figures
output_path = 'output/ASA/%s_%s/' % (feature_x, feature_y)
tfboard_path=None

# Set output label classes
label_cla = 'PM_Reversal'
label_fm = None

# Set data column
date_column = "smDate"

# Set security ID column
security_id = 'SecID'

# Set train and test period
train_begin = "1996-01-01"
train_end = "2020-04-30"
test_begin = "1996-01-01"
test_end = "2020-04-30"

# Cross validation configuration
train_from_future = True
force_val_length = False

# Prediction configuration
expand_training_window = False

# Set cross-validation configuration
k = 2     # Must be > 1
n_epoch = 1
subsample = 1
purge_length = 3

# Set p-value threshold for ANOVA test p_thres = 0.05
p_thres = 0.05

# Set metric for training
cv_metric = ['recall', 'f1-score', 'precision', 'accuracy']

# Set color scheme for decision boundary plot
cmap = matplotlib.cm.get_cmap('Reds')
#db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
db_colors = ['white', 'crimson']

N_CLASS=2
db_colors_scatter = ['white', 'crimson']

# Set algorithms to run
run_lr = True
run_xgb = False
run_nn = False
save_prediction = False

#-------------------------------------------------------------------------------
# Convert configuration as a dictionary
#-------------------------------------------------------------------------------
config = {
    'feature_x':feature_x, 'feature_y':feature_y,
    'output_path':output_path, 'security_id':security_id,
    'label_cla':label_cla, 'label_fm':label_fm,
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
    df_beta = pd.read_excel(beta_corr_path)

    # Discretize PM and DIFF
    df = utils.discretize_variables_by_month(
        df=df, variables=[PM, DIFF],
        n_classes=PM_N_CLASS, suffix="n%s" %PM_N_CLASS,
        class_names=\
            ["Q%d" %x for x in range(PM_N_CLASS,0,-1)],
        month=date_column)

    # Group by two variables and time
    df[return_group] = \
	df[feats_disc[0]].astype(str)\
	    .apply(lambda x:"%s %s" %(features[0][:-4], x))\
	+ df[feats_disc[1]].astype(str)\
	    .apply(lambda x:"; %s %s" %(features[1][:-4], x))

    # Plot relative return of each group
    df_reversal = df.groupby([pm_date, return_group])[FQ_REL_RETURN]\
		    .mean().unstack()\
		    .rolling(window=moving_average).mean()

    df_reversal = pd.DataFrame(
	    df_reversal['Diff Q5; PM Q5'] - df_reversal['Diff Q1; PM Q1'] > thres)\
	.rename({0:'PM_Reversal'}, axis=1)\
	.astype(int)

    # Join beta correlation and PM reversal
    df_beta = df_beta.set_index(date_column)\
        .join(df_reversal)

    # Calculate changes in feature x as feature y
    df_beta[feature_y] = df_beta[feature_x].diff(periods=dt)

    # Remove null from 1) dt calculation and 2) joining
    df_beta = df_beta.dropna().reset_index()

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df_beta, date_column, test_begin, test_end,
	train_begin=train_begin, train_end=train_end,
	train_from_future=train_from_future)

    #---------------------------------------------------------------------------
    # Logistic regression
    #---------------------------------------------------------------------------
    if run_lr:
        # Set parameters to search
        param_grid_lr = {
            "penalty":['none'],
            "multi_class":['multinomial'],
            "solver":['newton-cg'],
            "max_iter":[50],
            "tol": [1e-2],
            "n_jobs":[-1],
            #"C": [1] + np.logspace(-4, 4, 10), # C <= 1e-5 doesn't converge
            "C": [1],
            "fit_intercept": [True]}

        # Set model
        model_lr = LogisticRegression()
        model_lr_str = 'lr'

        # Run analysis on 2D decision boundary
        rs_lr = classification.decision_boundary2D(
            config, df_train, df_test,
            model_lr, model_lr_str, param_grid_lr, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=False,
            model_evaluation=True,
            plot_decision_boundary=True,
            save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-0.25,2.0))

    #---------------------------------------------------------------------------
    # Xgboost
    #---------------------------------------------------------------------------
    if run_xgb:
        # Set parameters to search
        param_grid_xgb = {
            #'min_child_weight': [1500, 1000, 500, 100], 
            #'min_child_weight': [1000, 500], 
            #'min_child_weight': [750, 500, 100, 50], 
            'min_child_weight': [0, 5, 10], 

            #'max_depth': [3, 10],
            'max_depth': [2, 3],

            'eta': [0.3], #[0.3]
            #'eta': [0.3, 0.6, 0.11, 0.01], #[0.3]

            #'n_estimators': [50],
            #'n_estimators': [50, 100],
            'n_estimators': [25],
        
            'gamma': [0], #[0, 5, 10, 20],
            #'gamma': [1,0.7,0.5,0.3],

            'lambda': [1],
            #'lambda': np.logspace(0, 2, 3), 

            'subsample': [1],
            #'subsample': [0.5, 0.8, 1],

            'n_jobs':[-1],
            'objective': ['multi:softmax'],
            'num_class': [2],
            }

        # Set model
        model_xgb = XGBClassifier()
        model_xgb_str = 'xgb'


        # Run analysis on 2D decision boundary
        rs_xgb = classification.decision_boundary2D(
            config, df_train, df_test,
            model_xgb, model_xgb_str, param_grid_xgb, best_params={},
            read_last=False,
            cv_study=True,
            run_backtest=False,
            model_evaluation=False,
            plot_decision_boundary=True,
            save_csv=True,
            return_train_ylim=(-1,20), return_test_ylim=(-0.25,2.0))


    #---------------------------------------------------------------------------
    # Neural net
    #---------------------------------------------------------------------------
    if run_nn:
        N_INPUT = 2
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Define ensemble of weak leaners
            ensemble = []
            input_layer = tf.keras.Input(shape=(N_INPUT,))
            for i in range(500):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=2, activation='softmax',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model0 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(N_INPUT,))
            for i in range(500):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.010),
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.010),
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=2, activation='softmax',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model1 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            ensemble = []
            input_layer = tf.keras.Input(shape=(N_INPUT,))
            for i in range(500):
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.050),
                    kernel_initializer=tf.initializers.GlorotUniform())(input_layer)
                weak_learner = tf.keras.layers.Dense(units=32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.050),
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner)
                ensemble.append(tf.keras.layers.Dense(units=2, activation='softmax',
                    kernel_initializer=tf.initializers.GlorotUniform())(weak_learner))
            output_layer = tf.keras.layers.Average()(ensemble)
            ensemble_model2 = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            #-------------------------------------------------------------------

            param_grid_nn = {
                #'learning_rate': [1e-4, 5e-4, 1e-3], #np.logspace(-4,-2,6),
                #'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5], #np.logspace(-4,-2,6),
                'learning_rate': [1e-3], #np.logspace(-4,-2,6),
                #'patience': [1, 3, 10], # 3,4,5
                'patience': [1], # 3,4,5
                'metrics': {
                    tf.keras.metrics.BinaryCrossentropy():'BinaryCrossEntropy'
                    },
                'loss': {
                    tf.keras.losses.BinaryCrossentropy():'BinaryCrossEntropy'
                    },
                'epochs': [1000],
                'validation_split': [0.2],
                'batch_size': [1024],
                'model': {
                    #ensemble_model0:'[32-32-1]*500_labmda0.005',
                    #ensemble_model1:'[32-32-1]*500_labmda0.010',
                    ensemble_model2:'[32-32-1]*500_labmda0.050',
                    #ensemble_model0_reg:'[32-32-1]*500(L2)',
                    #ensemble_model1:'[32-0.5-32-0.5-1]*100',
                    #ensemble_model1_reg:'[32-0.5-32-0.5-1]*100(L2)',
                    #ensemble_model2:'[32-0.5-32-0.5-32-0.5-1]*100',
                    #ensemble_model2_reg:'[32-0.5-32-0.5-32-0.5-1]*500(L2)',
                    #ensemble_model3:'[64-0.5-64-0.5-1]*100',
                    #ensemble_model4:'[64-0.5-64-0.5-64-0.5-1]*100',
                    #ensemble_model5:'[128-0.5-128-0.5-1]*100',
                    },
                }

            # Build model and evaluate
            model_nn = tfmodel.TensorflowModel(
                model=None, params={}, log_path=tfboard_path, model_type='cla')
            model_nn_str = 'nn'

            # Run analysis on 2D decision boundary
            rs_nn = classification.decision_boundary2D(
                config, df_train, df_test,
                model_nn, model_nn_str, param_grid_nn, best_params={},
                read_last=False,
                cv_study=True,
                run_backtest=False,
                model_evaluation=True,
                plot_decision_boundary=True,
                save_csv=True,
                return_train_ylim=(-1,20), return_test_ylim=(-0.25,2.0))



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







