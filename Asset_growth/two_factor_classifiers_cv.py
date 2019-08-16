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
from Asset_growth.lib.heuristicModel import *
from Asset_growth.lib.plots import *
from Asset_growth.lib.purged_k_fold import *
from Asset_growth.lib.stats import *
from Asset_growth.lib.utils import *



#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '/mnt/mainblob/asset_growth/data/\
Data_for_AssetGrowth_Context.r5.p2.csv'

# Set available features and labels
""" Available features:
'GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC',
'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum' """
feature_x = 'AG'
feature_y = 'FCFA'
features = [feature_x, feature_y]

# Set path to save output figures
output_path = 'plots/%s_%s/' % (feature_x, feature_y)

# Set labels
label = "fqTotalReturn_tertile"     # or "fmTotalReturn_quintile"
label_reg = "fqTotalReturn"         # or "fmTotalReturn"
label_fm = "fmTotalReturn"          # Used to calculate cumulative return
date_column = "eom"

# Set train and test period
test_begin = "2011-01-01"
test_end = "2017-11-01"

# Set k-fold
k = 10

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
run_summary = True



#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)

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

    # Assign tertile labels to the features
    df = discretize_variables_by_month(
        df=df, variables=features, month="eom",
        labels_tertile={feature_x:[2, 1, 0], feature_y:[2, 1, 0]}, # 0 is high
        labels_quintile={feature_x:[4, 3, 2, 1, 0],
                         feature_y:[4, 3, 2, 1, 0]}) # 0 is high
    tertile_boundary = get_tertile_boundary(df, features)

    # Split dataset into train and test dataset
    df_train, df_test = train_test_split_by_date(
        df, date_column, test_begin, test_end)

    # Create directory to save figures
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #---------------------------------------------------------------------------
    # Logistic regression
    #---------------------------------------------------------------------------
    if run_lr:
        print("Running logistic regression")
        # Set parameters to search
        param_grid = {
            "penalty":['l2'],
            "multi_class":['multinomial'],
            "solver":['newton-cg'],
            "max_iter":[100],
            "n_jobs":[-1],
            "C": np.logspace(-5, 1, 10)} #[1, 100]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            cv_results_lr = grid_search_purged_cv(
                df_train=df_train,
                model=LogisticRegression(),
                param_grid=param_grid,
                metric=cv_metric,
                n_epoch=10, subsample=0.8,
                features=features, label=label,
                k=k, purge_length=3,
                output_path=output_path+"lr/cross_validation/",
                verbose=False)
            # Perform ANOVA to select best model
            anova_results_lr = select_best_model_by_anova(
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
        plot_cv_dist(
            cv_results_lr,
            n_bins=10, x_range=None,
            legend_loc=None, legend_box=(1, 1), figsize=(18, 10), alpha=0.6,
            hist_type='stepfilled', edgecolor='black',
            filename=output_path+"lr/cross_validation/cv_hist")
        plot_cv_box(
            cv_results_lr,
            filename=output_path+"lr/cross_validation/cv_box",
            cv_metric=None, figsize=(18, 10), color="#3399FF")

        # Plot decision boundaries of all hyperparameter sets
        plot_decision_boundary_multiple_hparmas(
            param_grid=param_grid, label=label, model=LogisticRegression(),
            df=df_train, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y],
            colors=colors, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=[0, 1, 2],
            filename=output_path+"lr/decision_boundary/db")

        #-----------------------------------------------------------------------
        # Cumulative return
        #-----------------------------------------------------------------------
        # Calculate cumulative return using best parameters
        df_cum_train_lr, df_cum_test_lr, model_lr = \
            predict_and_calculate_cum_return(
                model=LogisticRegression(**best_params_lr),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_lr, df_cum_test_lr, label_reg,
            group_label={0:'T1', 1:'T2', 2:'T3'},
            figsize=(8, 6),
            filename=output_path+"lr/cum_return/return_by_group",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_lr], return_label=[0, 1, 2],
            list_labels=["lr"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path+"lr/cum_return/return_diff_group",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        #-----------------------------------------------------------------------
        # Decision boundary
        #-----------------------------------------------------------------------
        # Plot decision boundary of the trained model with best params.
        plot_decision_boundary(
            model=model_lr, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8), ticks=[0, 1, 2],
            annot={
                'text':str(best_params_lr).strip('{}')\
                    .replace('\'','')\
                    .replace(',','\n')\
                    .replace('\n ', '\n'),
                'x':0.02, 'y':0.98},
            filename=output_path+"lr/decision_boundary/db_best_model")


    #---------------------------------------------------------------------------
    # Xgboost
    #---------------------------------------------------------------------------
    if run_xgb:
        print("Running xgboost")
        # Set parameters to search
        param_grid = {
            'min_child_weight': [1000, 500, 100, 10],
            'max_depth': [1, 4, 5, 20],
            'learning_rate': [0.3],
            'n_estimators': [1],
            'objective': ['multi:softmax'],
            'gamma': [10.0], #np.logspace(-2, 1, 1), # Min loss reduction
            'lambda': [1], #np.logspace(0, 2, 2) # L2 regularization
            'n_jobs':[-1],
            'num_class': [3]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            cv_results_xgb = grid_search_purged_cv(
                df_train=df_train,
                model=XGBClassifier(),
                param_grid=param_grid,
                metric=cv_metric,
                n_epoch=10, subsample=0.8,
                features=features, label=label,
                output_path=output_path+"xgb/cross_validation/",
                k=k, purge_length=3, verbose=False)
            # Perform ANOVA to select best model
            anova_results_xgb = select_best_model_by_anova(
                cv_results=cv_results_xgb,
                cv_metric=cv_metric,
                param_grid=param_grid, p_thres=p_thres)
            # Get best parameters
            best_params_xgb = anova_results_xgb['best_params']
        else:
            best_params_xgb = {
                'max_depth': 5, 'learning_rate': 0.3, 'n_estimators': 50,
                'objective': 'multi:softmax', 'min_child_weight': 1000.0,
                'gamma': 10.0, 'lambda': 1, 'subsample': 0.5, 'n_jobs': -1,
                'num_class': 3}


        #-----------------------------------------------------------------------
        # Cross-validation results
        #-----------------------------------------------------------------------
        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_xgb,
            n_bins=10, x_range=None,
            legend_loc=None, legend_box=(1, 1), figsize=(18, 10), alpha=0.6,
            hist_type='stepfilled', edgecolor='black',
            filename=output_path+"xgb/cross_validation/cv_hist")
        plot_cv_box(
            cv_results_xgb,
            filename=output_path+"xgb/cross_validation/cv_box",
            cv_metric=None, figsize=(18, 10), color="#3399FF")

        # Plot decision boundaries of all hyperparameter sets
        plot_decision_boundary_multiple_hparmas(
            param_grid=param_grid, label=label, model=XGBClassifier(),
            df=df_train, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y],
            colors=colors, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=[0, 1, 2],
            filename=output_path+"xgb/decision_boundary/db")

        #-----------------------------------------------------------------------
        # Cumulative return
        #-----------------------------------------------------------------------
        # Calculate cumulative return using best parameters
        df_cum_train_xgb, df_cum_test_xgb, model_xgb = \
            predict_and_calculate_cum_return(
                model=XGBClassifier(**best_params_xgb),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_xgb, df_cum_test_xgb, label_reg,
            group_label={0:'T1', 1:'T2', 2:'T3'},
            figsize=(8, 6),
            filename=output_path+"xgb/cum_return/return_by_group",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_xgb], return_label=[0, 1, 2],
            list_labels=["xgb"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path+"xgb/cum_return/return_diff_group",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        #-----------------------------------------------------------------------
        # Decision boundary
        #-----------------------------------------------------------------------
        # Plot decision boundary of the trained model with best params.
        plot_decision_boundary(
            model=model_xgb, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            annot={
                'text':str(best_params_xgb).strip('{}')\
                    .replace('\'','')\
                    .replace(',','\n')\
                    .replace('\n ', '\n'),
                'x':0.02, 'y':0.98},
            filename=output_path+"xgb/decision_boundary/db_best_model")


    #---------------------------------------------------------------------------
    # SVM
    #---------------------------------------------------------------------------
    if run_svm:
        print("Running svm")
        # Set parameters to search
        param_grid =  {
            'C': np.logspace(-3, 2, 5),
            'penalty': ['l2']
            }

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            cv_results_svm = grid_search_purged_cv(
                df_train=df_train,
                model=LinearSVC(),
                param_grid=param_grid,
                metric=cv_metric,
                n_epoch=10, subsample=0.8,
                features=features, label=label,
                output_path=output_path+"svm/cross_validation/",
                k=k, purge_length=3, verbose=False)
            # Perform ANOVA to select best model
            anova_results_svm = select_best_model_by_anova(
                cv_results=cv_results_svm,
                cv_metric=cv_metric,
                param_grid=param_grid, p_thres=p_thres)
            # Get best parameters
            best_params_svm = anova_results_svm['best_params']
        else:
            best_params_svm = {}


        #-----------------------------------------------------------------------
        # Cross-validation results
        #-----------------------------------------------------------------------
        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_svm,
            n_bins=10, x_range=None,
            legend_loc=None, legend_box=(1, 1), figsize=(18, 10), alpha=0.6,
            hist_type='stepfilled', edgecolor='black',
            filename=output_path+"svm/cross_validation/cv_hist")
        plot_cv_box(
            cv_results_svm,
            filename=output_path+"svm/cross_validation/cv_box",
            cv_metric=None, figsize=(18, 10), color="#3399FF")

        # Plot decision boundaries of all hyperparameter sets
        plot_decision_boundary_multiple_hparmas(
            param_grid=param_grid, label=label, model=LinearSVC(),
            df=df_train, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y],
            colors=colors, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=[0, 1, 2],
            filename=output_path+"svm/decision_boundary/db")

        #-----------------------------------------------------------------------
        # Cumulative return
        #-----------------------------------------------------------------------
        # Calculate cumulative return using best parameters
        df_cum_train_svm, df_cum_test_svm, model_svm = \
            predict_and_calculate_cum_return(
                model=LinearSVC(**anova_results_svm),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_svm, df_cum_test_svm, label_reg,
            group_label={0:'T1', 1:'T2', 2:'T3'},
            figsize=(8, 6),
            filename=output_path+"svm/cum_return/return_by_group",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_svm], return_label=[0, 1, 2],
            list_labels=["svm"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path+"svm/cum_return/return_diff_group",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        #-----------------------------------------------------------------------
        # Decision boundary
        #-----------------------------------------------------------------------
        # Plot decision boundary of the trained model with best params.
        plot_decision_boundary(
            model=model_svm, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            annot={
                'text':str(best_params_svm).strip('{}')\
                    .replace('\'','')\
                    .replace(',','\n')\
                    .replace('\n ', '\n'),
                'x':0.02, 'y':0.98},
            filename=output_path+"svm/decision_boundary/db_best_model")

    #---------------------------------------------------------------------------
    # kNN
    #---------------------------------------------------------------------------
    if run_knn:
        print("Running kNN")
        # Set parameters to search
        param_grid = {'n_neighbors': [int(x) for x in np.logspace(1, 8, 10)]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            cv_results_knn = grid_search_purged_cv(
                df_train=df_train,
                model=KNeighborsClassifier(),
                param_grid=param_grid,
                metric=cv_metric,
                n_epoch=10, subsample=0.8,
                features=features, label=label,
                output_path=output_path+"knn/cross_validation/",
                k=k, purge_length=3, verbose=False)
            # Perform ANOVA to select best model
            anova_results_knn = select_best_model_by_anova(
                cv_results=cv_results_knn,
                cv_metric=cv_metric,
                param_grid=param_grid, p_thres=p_thres)
            # Get best parameters
            best_params_knn = anova_results_knn['best_params']
        else:
            best_params_knn = {'n_neighbors': [10000]}


        #-----------------------------------------------------------------------
        # Cross-validation results
        #-----------------------------------------------------------------------
        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_knn,
            n_bins=10, x_range=None,
            legend_loc=None, legend_box=(1, 1), figsize=(18, 10), alpha=0.6,
            hist_type='stepfilled', edgecolor='black',
            filename=output_path+"knn/cross_validation/cv_hist")
        plot_cv_box(
            cv_results_knn,
            filename=output_path+"knn/cross_validation/cv_box",
            cv_metric=None, figsize=(18, 10), color="#3399FF")

        # Plot decision boundaries of all hyperparameter sets
        plot_decision_boundary_multiple_hparmas(
            param_grid=param_grid, label=label, model=KNeighborsClassifier(),
            df=df_train, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y],
            colors=colors, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=[0, 1, 2],
            filename=output_path+"knn/decision_boundary/db")

        #-----------------------------------------------------------------------
        # Cumulative return
        #-----------------------------------------------------------------------
        # Calculate cumulative return using best parameters
        df_cum_train_knn, df_cum_test_knn, model_knn = \
            predict_and_calculate_cum_return(
                model=KNeighborsClassifier(**best_params_knn),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_knn, df_cum_test_knn, label_reg,
            group_label={0:'T1', 1:'T2', 2:'T3'},
            figsize=(8, 6),
            filename=output_path+"knn/cum_return/return_by_group",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_knn], return_label=[0, 1, 2],
            list_labels=["knn"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path+"knn/cum_return/return_diff_group",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        #-----------------------------------------------------------------------
        # Decision boundary
        #-----------------------------------------------------------------------
        # Plot decision boundary of the trained model with best params.
        plot_decision_boundary(
            model=model_knn, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            annot={
                'text':str(best_params_knn).strip('{}')\
                    .replace('\'','')\
                    .replace(',','\n')\
                    .replace('\n ', '\n'),
                'x':0.02, 'y':0.98},
            filename=output_path+"knn/decision_boundary/db_best_model")
        
    #---------------------------------------------------------------------------
    # Classification by simple sort
    #---------------------------------------------------------------------------
    if run_sort:
        print("Running sort by %s" % feature_x)
        # Calculate cumulative return
        df_cum_train_sort_x = cumulative_return(
            df=df_train, var_classes='%s_tertile' % feature_x,
            total_return=label_fm)
        df_cum_test_sort_x = cumulative_return(
            df=df_test, var_classes='%s_tertile' % feature_x,
            total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_sort_x = df_cum_train_sort_x.rename(
            {'%s_tertile' % feature_x:"pred"}, axis=1)
        df_cum_test_sort_x = df_cum_test_sort_x.rename(
            {'%s_tertile' % feature_x:"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_sort_x, df_cum_test_sort_x, label_reg, figsize=(8, 6),
            group_label={
                2: '%s low' % feature_x, 
                1: '%s mid' % feature_x,
                0: '%s high' % feature_x},
            kwargs_train={'ylim':(-1, 7)},
            kwargs_test={'ylim':(-1, 5)}, filename=output_path+"return_sort")

        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_sort_x],
            list_labels=["Sort %s" % feature_x], label_reg=label_reg,
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)},
            # For special case of AG, change it to [2, 1, 0]
            # so that it calculates Low AG - High AG.
            return_label=[0, 1, 2], 
            figsize=(8, 6), filename=output_path+"return_sort_diff")

        print("Running sort by %s" %feature_y)

        # Calculate cumulative return
        df_cum_train_sort_y = cumulative_return(
            df=df_train, var_classes='%s_tertile' % feature_y,
            total_return=label_fm)
        df_cum_test_sort_y = cumulative_return(
            df=df_test, var_classes='%s_tertile' % feature_y,
            total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_sort_y = df_cum_train_sort_y.rename(
            {'%s_tertile' % feature_y:"pred"}, axis=1)
        df_cum_test_sort_y = df_cum_test_sort_y.rename(
            {'%s_tertile' % feature_y:"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_sort_y, df_cum_test_sort_y, label_reg, figsize=(8, 6),
            group_label={
                2: '%s low' % feature_y,
                1: '%s mid' % feature_y,
                0: '%s high' % feature_y},
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 5)},
            filename=output_path+"return_sort_%s" % feature_y)

        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_sort_y],
            list_labels=["Sort %s" % feature_y], label_reg=label_reg,
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)},
            return_label=[0, 1, 2],
            figsize=(8, 6),
            filename=output_path+"return_sort_diff_%s" % feature_y)




    #---------------------------------------------------------------------------
    # Summary plots
    #---------------------------------------------------------------------------
    if run_summary:
        # Read Tensorflow results
        #df_cum_test_tf = pickle.load(
        #    open(output_path+'model_comparison/df_cum_test_tf.pickle', 'rb'))


        plot_cumulative_return_diff(
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
        save_summary(
            df_train, label, cv_metric, output_path+"model_comparison/",
            cv_results={
                'Logistic': cv_results_lr,
                'XGB': cv_results_xgb,
                'KNN': cv_results_knn,
                'SVM': cv_results_svm,
            })



    print("Successfully completed all tasks")


