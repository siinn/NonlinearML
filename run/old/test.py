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
from sklearn.neural_network import MLPClassifier
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


#-------------------------------------------------------------------------------
# Set user options
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
features = [feature_x, feature_y]

# Set path to save output figures
output_path = 'plots/%s_%s/' % (feature_x, feature_y)

# Set labels
label_reg = "fqTotalReturn"         # or "fmTotalReturn"
label_fm = "fmTotalReturn"          # Used to calculate cumulative return
date_column = "eom"

# Set number of output label classes
n_classes=3
class_label={0:'T1', 1:'T2', 2:'T3'}
suffix="descrite"
label = "_".join([label_reg, suffix])

# Set train and test period
test_begin = "2011-01-01"
test_end = "2017-11-01"

# Set cross-validation configuration
k = 2
n_epoch = 10
subsample = 0.8
purge_length = 3

# Set p-value threshold for ANOVA test
p_thres = 0.05

# Set metric for training
cv_metric = 'f1-score'

# Set color scheme for decision boundary plot
colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]

# Set algorithms to run
run_grid_search = True
run_lr = True
run_xgb = True
run_svm = True
run_knn = True
run_sort = False
run_summary = False





#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    print('Running two factor classification with the following two factors:')
    print(' > feature x: %s' % feature_x)
    print(' > feature y: %s' % feature_y)

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(INPUT_PATH, index_col=[0], parse_dates=[date_column])

    # Discretize label
    df = utils.discretize_variables_by_month(
        df=df, variables=[label_reg], month=date_column, n_classes=n_classes,
        class_names=sorted(np.arange(n_classes), reverse=True),
        suffix=suffix)

    #tertile_boundary = utils.get_tertile_boundary(df, features)

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end)

    #---------------------------------------------------------------------------
    # Logistic regression
    #---------------------------------------------------------------------------
    if run_lr:
        print("Running logistic regression")

        # Set output path for this model
        output_path_model = output_path + "lr/"

        # Set parameters to search
        param_grid = {
            "penalty":['l2'],
            "multi_class":['multinomial'],
            "solver":['newton-cg'],
            "max_iter":[1],
            "n_jobs":[-1],
            "C": np.logspace(-5, 1, 1)} #[1, 100]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            cv_results_lr = cv.grid_search(
                df_train=df_train,
                model=LogisticRegression(),
                param_grid=param_grid,
                metric=cv_metric,
                n_epoch=n_epoch, subsample=subsample,
                features=features, label=label,
                k=k, purge_length=purge_length,
                output_path=output_path_model+"cross_validation/",
                verbose=False)
            # Perform ANOVA to select best model
            anova_results_lr = stats.select_best_model_by_anova(
                cv_results=cv_results_lr,
                cv_metric=cv_metric,
                param_grid=param_grid, p_thres=p_thres)
            # Get best parameters
            best_params_lr = anova_results_lr['best_params']
        else:
            best_params_lr = {}

        #-----------------------------------------------------------------------
        # Cross-validation results
        #-----------------------------------------------------------------------
        # Plot distribution of cross-validation results
        plot_cv.plot_cv_dist(
            cv_results_lr,
            n_bins=10, x_range=None,
            legend_loc=None, legend_box=(1, 1), figsize=(18, 10), alpha=0.6,
            hist_type='stepfilled', edgecolor='black',
            filename=output_path_model+"cross_validation/cv_hist")
        plot_cv.plot_cv_box(
            cv_results_lr,
            filename=output_path_model+"cross_validation/cv_box",
            cv_metric=None, figsize=(18, 10), color="#3399FF")

        # Plot decision boundaries of all hyperparameter sets
        plot_db.decision_boundary_multiple_hparmas(
            param_grid=param_grid, label=label, model=LogisticRegression(),
            df=df_train, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            #vlines=tertile_boundary[feature_x],
            #hlines=tertile_boundary[feature_y],
            colors=colors, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=sorted(np.arange(n_classes), reverse=True),
            filename=output_path_model+"decision_boundary/db")

        #-----------------------------------------------------------------------
        # Cumulative return
        #-----------------------------------------------------------------------
        # Calculate cumulative return using best parameters
        df_cum_train_lr, df_cum_test_lr, model_lr = \
            utils.predict_and_calculate_cum_return(
                model=LogisticRegression(**best_params_lr),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)

        # Make cumulative return plot
        plot_backtest.plot_cumulative_return(
            df_cum_train_lr, df_cum_test_lr, label_reg,
            group_label=class_label,
            figsize=(8, 6),
            filename=output_path_model+"cum_return/return_by_group",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_backtest.plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_lr], return_label=[0, 1, 2],
            list_labels=["lr"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path_model+"cum_return/return_diff_group",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        #-----------------------------------------------------------------------
        # Decision boundary
        #-----------------------------------------------------------------------
        # Plot decision boundary of the trained model with best params.
        plot_db.decision_boundary(
            model=model_lr, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            #vlines=tertile_boundary[feature_x],
            #hlines=tertile_boundary[feature_y],
            colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=sorted(np.arange(n_classes), reverse=True),
            annot={
                'text':str(best_params_lr).strip('{}')\
                    .replace('\'','')\
                    .replace(',','\n')\
                    .replace('\n ', '\n'),
                'x':0.02, 'y':0.98},
            filename=output_path_model+"decision_boundary/db_best_model")













    #---------------------------------------------------------------------------
    # Summary plots
    #---------------------------------------------------------------------------
    if run_summary:
        # Read Tensorflow results
        #df_cum_test_tf = pickle.load(
        #    open(output_path+'model_comparison/df_cum_test_tf.pickle', 'rb'))


        plot_backtest.plot_cumulative_return_diff(
            list_cum_returns=[
                df_cum_test_lr, df_cum_test_knn,
                df_cum_test_xgb, df_cum_test_svm],
            list_labels=["Linear", "KNN", "XGB", "SVM"],
            label_reg=label_reg,
            figsize=(8, 6), return_label=[0, 1, 2],
            kwargs_train={'ylim':(-1, 3)},
            kwargs_test={'ylim':(-1, 3)},
            legend_order=["XGB", "Linear", "KNN", "SVM"],
            filename=output_path+"model_comparison/return_diff_summary")

        # Save results
        utils.save_summary(
            df_train, label, cv_metric, output_path+"model_comparison/",
            cv_results={
                'Logistic': cv_results_lr,
                'XGB': cv_results_xgb,
                'KNN': cv_results_knn,
                'SVM': cv_results_svm,
            })



    print("Successfully completed all tasks")


