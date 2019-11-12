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
#db_xlim = (-3, 3)
#db_ylim = (-3, 3)
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
#db_xlim = (-3, 3)
#db_ylim = (-3, 3)
#db_res = 0.01





# Set path to save output figures
output_path = 'output/%s_%s/reg/' % (feature_x, feature_y)
tfboard_path='tf_log/%s_%s/' % (feature_x, feature_y)

# Set labels
n_classes=10
class_label={x+1:'D'+str(x+1) for x in range(10)}
class_order = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] # High return to low return

# Set output label classes
label_reg = 'fqRet' # continuous target label
label_cla = 'QntfqRet' # discretized target label
label_fm = 'fmRet' # monthly return used for calculating cum. return

# Set data column
date_column = "smDate"

# Set security ID column
security_id = 'SecurityID'

# Set train and test period
test_begin = "2012-01-01"
test_end = "2019-05-01"

# Set cross-validation configuration
k = 5            # Must be > 1
n_epoch = 1
subsample = 0.5
purge_length = 3

# Set p-value threshold for ANOVA test p_thres = 0.05
p_thres = 0.05

# Set metric for training
cv_metric = ['f1-score', 'precision', 'recall', 'accuracy']

# Set color scheme for decision boundary plot
cmap = matplotlib.cm.get_cmap('Spectral', 10)
db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# Set decision boundary plotting options
db_figsize= (8, 8)
db_annot_x=0.02
db_annot_y=0.98
db_nbins=50

# Set algorithms to run
run_comparison = True
save_prediction = True

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

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end)



    #---------------------------------------------------------------------------
    # Compare model results
    #---------------------------------------------------------------------------
    models = ['lr', 'knn', 'xgb']
    if run_comparison:
        model_summary = summary.model_comparison(
            models=models,
            output_path=output_path,
            label_reg=config['label_reg'],
            class_label=config['class_order'],
            date_column=config['date_column'],
            col_pred='pred_rank',
            ylim=(-0.5,0.8))


    #---------------------------------------------------------------------------
    # Concatenate predictions to original date
    #---------------------------------------------------------------------------
    if save_prediction:
        predictions = summary.save_prediction(
            models=models,
            feature_x=config['feature_x'],
            feature_y=config['feature_y'],
            df_input=df,
            output_path=output_path,
            date_column=config['date_column'])


    #---------------------------------------------------------------------------
    # Print number of predictions
    #---------------------------------------------------------------------------


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

    







