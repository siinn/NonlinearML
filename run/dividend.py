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
import warnings
from xgboost.sklearn import XGBClassifier

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.utils as utils
import NonlinearML.lib.stats as stats
import NonlinearML.lib.io as io

import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.cross_validation as plot_cv

import NonlinearML.interprete.classification as classification


#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/Data_EM.csv'

# Set features of interest
feature_x = 'DividendYield'
feature_y = 'Payout_E'

# Set path to save output figures
output_path = 'output/%s_%s/' % (feature_x, feature_y)

# Set labels
n_classes=10
class_label={x+1:'D'+str(x+1) for x in range(10)}

# Set output label classes
label_reg = 'fqRet' # continuous target label
label_cla = 'QntfqRet' # discretized target label
label_fm = 'fmRet' # monthly return used for calculating cum. return

# Set data column
date_column = "smDate"

# Set train and test period
test_begin = "2012-01-01"
test_end = "2019-05-01"

# Set cross-validation configuration
k = 5           # Must be > 1
n_epoch = 5
subsample = 0.8
purge_length = 3

# Set p-value threshold for ANOVA test
p_thres = 0.05

# Set metric for training
cv_metric = 'f1-score'

# Set color scheme for decision boundary plot
#db_colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]
cmap = matplotlib.cm.get_cmap('Spectral', 10)
db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

# Set algorithms to run
run_lr = False
run_xgb = True
run_comparison = True

# Set default warning
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'

#-------------------------------------------------------------------------------
# Convert configuration as a dictionary
#-------------------------------------------------------------------------------
config = {
    'feature_x':feature_x, 'feature_y':feature_y,
    'output_path':output_path,
    'n_classes':n_classes, 'class_label':class_label,
    'label_reg':label_reg, 'label_cla':label_cla, 'label_fm':label_fm,
    'date_column':date_column, 'test_begin':test_begin, 'test_end':test_end,
    'k':k, 'n_epoch':n_epoch, 'subsample':subsample,
    'purge_length':purge_length,
    'p_thres':p_thres, 'cv_metric':cv_metric, 'db_colors':db_colors}


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":


    io.title('Running two factor classification with two factors:')
    print(' > feature x: %s' % config['feature_x'])
    print(' > feature y: %s' % config['feature_y'])

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(INPUT_PATH, index_col=None, parse_dates=[date_column])

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
            "C": np.logspace(-4, 3, 10)} # note: C <= 1e-5 doesn't converge

        # Set model
        model_lr = LogisticRegression()
        model_lr_str = 'lr'

        # Run analysis on 2D decision boundary
        db_lr = classification.decision_boundary2D(
            config, df_train, df_test,
            model_lr, model_lr_str, param_grid_lr, best_params={},
            cv_study=True, calculate_return=True,
            plot_decision_boundary=True, save_csv=True)

    #---------------------------------------------------------------------------
    # Xgboost
    #---------------------------------------------------------------------------
    if run_xgb:
        # Set parameters to search
        param_grid_xgb = {
            'min_child_weight': [1000, 500],
            'max_depth': [5, 20, 50],
            'learning_rate': [0.1],
            'n_estimators': [50],
            'objective': ['multi:softmax'],
            'gamma': [0], #np.logspace(-2, 1, 1), # Min loss reduction
            'lambda': [1], #np.logspace(0, 2, 2) # L2 regularization
            'n_jobs':[-1],
            'num_class': [n_classes]}

        # Set model
        model_xgb = XGBClassifier()
        model_xgb_str = 'xgb'

        # Run analysis on 2D decision boundary
        db_xgb = classification.decision_boundary2D(
            config, df_train, df_test,
            model_xgb, model_xgb_str, param_grid_xgb, best_params={},
            grid_search=True, cv_results=None,
            cv_study=True, calculate_return=True,
            plot_decision_boundary=True, save_csv=True,
            db_xlim=(0,0.2), db_ylim=(-1,1.5), db_res=0.001,
            return_train_ylim=(-1,20), return_test_ylim=(-1,5))

    #---------------------------------------------------------------------------
    # Compare model results
    #---------------------------------------------------------------------------
    if run_comparison:
        utils.model_comparison(
            models=['lr', 'xgb'], output_path=output_path,
            class_label=sorted(
                list(config['class_label'].keys()), reverse=True),
            date_column=config['date_column'])

    print("Successfully completed all tasks")


