#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.utils as utils
import NonlinearML.lib.stats as stats

import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.cross_validation as plot_cv

import NonlinearML.interprete.classification as classification

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
output_path = 'plots/%s_%s/' % (feature_x, feature_y)

# Set labels
n_classes=3
class_label={0:'T1', 1:'T2', 2:'T3'}
suffix="descrite"

# Set output label classes
label_reg = "fqTotalReturn" # continuous target label
label_cla = "_".join([label_reg, suffix]) # discretized target label
label_fm = "fmTotalReturn" # monthly return used for calculating cum. return

# Set data column
date_column = "eom"

# Set train and test period
test_begin = "2011-01-01"
test_end = "2017-11-01"

# Set cross-validation configuration
k = 10
n_epoch = 100
subsample = 0.8
purge_length = 3

# Set p-value threshold for ANOVA test
p_thres = 0.05

# Set metric for training
cv_metric = 'f1-score'

# Set color scheme for decision boundary plot
db_colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]

# Set algorithms to run
run_lr = True
run_xgb = True
run_svm = True
run_knn = True
run_summary = True


#-------------------------------------------------------------------------------
# Convert configuration as a dictionary
#-------------------------------------------------------------------------------
config = {
    'feature_x':feature_x, 'feature_y':feature_y,
    'output_path':output_path,
    'n_classes':n_classes, 'class_label':class_label, 'suffix':suffix,
    'label_reg':label_reg, 'label_cla':label_cla, 'label_fm':label_fm,
    'date_column':date_column, 'test_begin':test_begin, 'test_end':test_end,
    'k':k, 'n_epoch':n_epoch, 'subsample':subsample,
    'purge_length':purge_length,
    'p_thres':p_thres, 'cv_metric':cv_metric, 'db_colors':db_colors}


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    print('Running two factor classification with the following two factors:')
    print(' > feature x: %s' % config['feature_x'])
    print(' > feature y: %s' % config['feature_y'])

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(INPUT_PATH, index_col=[0], parse_dates=[date_column])

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
            "max_iter":[100],
            "n_jobs":[-1],
            "C": np.logspace(-5, 1e3, 10)} #[1, 100]}

        # Set model
        model_lr = LogisticRegression()
        model_lr_str = 'lr'

        # Run analysis on 2D decision boundary
        db_lr = classification.decision_boundary2D(
            config, df_train, df_test,
            model_lr, model_lr_str, param_grid_lr, best_params={},
            cv_study=True, calculate_return=True, plot_decision_boundary=True)

    #---------------------------------------------------------------------------
    # Xgboost
    #---------------------------------------------------------------------------
    if run_xgb:
        # Set parameters to search
        param_grid_xgb = {
            'min_child_weight': [1000, 500, 100, 10],
            'max_depth': [1, 4, 5, 20],
            #'min_child_weight': [1000],
            #'max_depth': [1],
            'learning_rate': [0.3],
            'n_estimators': [100],
            'objective': ['multi:softmax'],
            'gamma': [10.0], #np.logspace(-2, 1, 1), # Min loss reduction
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
            cv_study=True, calculate_return=True, plot_decision_boundary=True)

    #---------------------------------------------------------------------------
    # SVM
    #---------------------------------------------------------------------------
    if run_svm:
        # Set parameters to search
        param_grid_svm = {
            'C': np.logspace(-3, 2, 10),
            'penalty': ['l2']}

        # Set model
        model_svm = LinearSVC()
        model_svm_str = 'svm'

        # Run analysis on 2D decision boundary
        db_svm = classification.decision_boundary2D(
            config, df_train, df_test,
            model_svm, model_svm_str, param_grid_svm, best_params={},
            cv_study=True, calculate_return=True, plot_decision_boundary=True)

    #---------------------------------------------------------------------------
    # kNN
    #---------------------------------------------------------------------------
    if run_knn:
        # Set parameters to search
        param_grid_knn = {
            'n_neighbors': [int(x) for x in np.logspace(1, 8, 10)]}

        # Set model
        model_knn = KNeighborsClassifier()
        model_knn_str = 'knn'

        # Run analysis on 2D decision boundary
        db_knn = classification.decision_boundary2D(
            config, df_train, df_test,
            model_knn, model_knn_str, param_grid_knn, best_params={},
            cv_study=True, calculate_return=True, plot_decision_boundary=True)


    #---------------------------------------------------------------------------
    # Summary plots
    #---------------------------------------------------------------------------
    if run_summary:
        # Read Tensorflow results
        #df_cum_test_tf = pickle.load(
        #    open(output_path+'model_comparison/df_cum_test_tf.pickle', 'rb'))


        plot_backtest.plot_cumulative_return_diff(
            list_cum_returns=[
                db_lr['cum_return_test'],
                db_knn['cum_return_test'],
                db_xgb['cum_return_test'],
                db_svm['cum_return_test'],
                ],
            list_labels=["Linear", "KNN", "XGB", "SVM"],
            label_reg=label_reg,
            figsize=(8, 6), return_label=sorted(np.arange(n_classes)),
            kwargs_train={'ylim':(-1, 3)},
            kwargs_test={'ylim':(-1, 3)},
            legend_order=["XGB", "Linear", "KNN", "SVM"],
            filename=output_path+"model_comparison/return_diff_summary")

        # Save results
        utils.save_summary(
            df_train, label_cla, cv_metric, output_path+"model_comparison/",
            cv_results={
                'Logistic': db_lr['cv_results'],
                'XGB': db_xgb['cv_results'],
                'KNN': db_knn['cv_results'],
                'SVM': db_svm['cv_results'],
            })


    print("Successfully completed all tasks")


