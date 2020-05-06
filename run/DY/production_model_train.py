#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
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

import NonlinearML.xgb.objective as xgb_obj
import NonlinearML.xgb.metric as xgb_metric
import NonlinearML.interpret.regression as regression

# Supress warnings
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

# Set number of bins for ranking
rank_n_bins=10
rank_label={x:'D'+str(x) for x in range(rank_n_bins)}
rank_order=[0,1,2,3,4,5,6,7,8,9] # low to high return
rank_top = 9
rank_bottom = 0

# Set output label classes
label_reg = 'fqRet'             # continuous target label
label_cla = 'fqRet_discrete'    # discretized target label
label_fm = 'fmRet'              # monthly return. used for backtesting

# Winsorize label, followed by standardization with median in each month
standardize_label = True
winsorize_lower = 0.01
winsorize_upper = 0.99

# Set date column
date_column = "smDate"

# Set security ID column
security_id = 'SecurityID'

# Set train
train_begin = "1996-12-01"
train_end = "2019-06-30"

# Set test period
""" In training production model, test-begin and test-end dates are not used.
You can set them to training period, and it wouldn't change model training."""
test_begin, test_end = train_begin, train_end

# Set path to save output figures
output_path = 'output/DY/%s_%s/production/' % (feature_x, feature_y)

# Set path to save model
model_tag = 'v2'
model_path = 'output/DY/%s_%s/production/xgb/model/%s/' % (feature_x, feature_y, model_tag)

#-------------------------------------------------------------------------------
# Cross-validation configuration
#-------------------------------------------------------------------------------
# Prediction configuration
expand_training_window = False

# Cross validation configuration
train_from_future = True
force_val_length = False

# Set cross-validation configuration
k = 2    # Must be > 1
n_epoch = 1
subsample = 0.3
purge_length = 3

# Set p-value threshold for ANOVA test p_thres = 0.05
p_thres = 0.20

# Set metric for training
cv_metric = ['Top-Bottom-std', 'Top-Bottom', 'r2', 'mape']


#-------------------------------------------------------------------------------
# Plot configuration
#-------------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
    # Split dataset into train and test dataset
    #---------------------------------------------------------------------------
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end,
        train_begin, train_end,
        train_from_future=train_from_future)


    #---------------------------------------------------------------------------
    # Train model
    #---------------------------------------------------------------------------
    # Set parameters to search
    param_grid_xgb = {
        'min_child_weight': [1500], 
        'max_depth': [3],
        'eta': [0.3],
        'gamma': [5],
        'lambda': [1],
        'n_estimators': [50],
        'n_jobs':[-1],
        'objective':['reg:squarederror'],
        'subsample': [1],
        }
    # Set model
    model_xgb = XGBRegressor()
    model_xgb_str = 'xgb/'

    # Run analysis on 2D decision boundary
    rs_xgb = regression.regression_surface2D(
        config, df_train, df_test,
        model_xgb, model_xgb_str, param_grid_xgb, best_params={},
        read_last=False,
        cv_study=False,
        run_backtest=True,
        model_evaluation=True,
        plot_decision_boundary=True,
        plot_residual=True,
        save_csv=True,
        save_col=[config['security_id']],
        verbose=True)

    # Dump model as pickle
    utils.create_folder(model_path+"xgb.pickle")
    rs_xgb['model'].save_model(model_path+"xgb.%s" % model_tag)


    print("Successfully completed all tasks")
