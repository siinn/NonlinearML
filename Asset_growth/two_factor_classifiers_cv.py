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
from xgboost.sklearn import XGBClassifier

# Import custom libraries
from Asset_growth.lib.plots import *
from Asset_growth.lib.utils import *
from Asset_growth.lib.purged_k_fold import *
from Asset_growth.lib.heuristicModel import *

#----------------------------------------------
# Set user options
#----------------------------------------------
# Set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.p2.csv'

# Set available features and labels
''' Available features: 'GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC', 'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum' '''
feature_x = 'ROIC'
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
k = 5

# Set metric for training
metric = 'f1-score'

# Set color scheme for decision boundary plot
colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]

# Set algorithms to run
run_grid_search     = True
run_lr              = True
run_xgb             = True
run_knn             = True
run_nn              = False
run_sort            = True
run_summary         = True



#----------------------------------------------
# Create output folder
#----------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)

#------------------------------------------
# Main 
#------------------------------------------
if __name__ == "__main__":

    print('Running two factor classification with the following two factors:')
    print(' > feature x: %s' % feature_x)
    print(' > feature y: %s' % feature_y)

    #------------------------------------------
    # Read dataset
    #------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=[0], parse_dates=[date_column])

    # Assign tertile labels to the features
    df = discretize_variables_by_month(df=df, variables=features, month="eom",
                                       labels_tertile={feature_x:[2,1,0], feature_y:[2,1,0]}, # 0 is high
                                       labels_quintile={feature_x:[4,3,2,1,0], feature_y:[4,3,2,1,0]}) # 0 is high
    tertile_boundary =  get_tertile_boundary(df, features)

    # Split dataset into train and test dataset
    df_train, df_test = train_test_split_by_date(df, date_column, test_begin, test_end)

    # Create directory to save figures
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #------------------------------------------
    # Logistic regression
    #------------------------------------------
    if run_lr:
        print("Running logistic regression")
        # Set parameters to search
        param_grid = {"penalty":['l2'],
                  "multi_class":['multinomial'],
                  "solver":['newton-cg'],
                  "max_iter":[100],
                  "n_jobs":[-1],
                  "C": np.logspace(-5,-1,10)} #[1,100]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_lr, cv_results_lr = grid_search_purged_cv(df_train=df_train,
                                                           model=LogisticRegression(),
                                                           param_grid=param_grid,
                                                           metric=metric,
                                                           features=features, label=label,
                                                           k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_lr.to_csv(output_path+"cv_results_lr.csv")
        else:
            best_params_lr = {'penalty': 'l2', 'multi_class': 'multinomial', 'solver': 'newton-cg', 'max_iter': 100, 'n_jobs': -1, 'C': 0.001}

        # Calculate cumulative return using best parameters
        df_cum_train_lr, df_cum_test_lr, model_lr = predict_and_calculate_cum_return(model=LogisticRegression(**best_params_lr),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_lr, df_cum_test_lr, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename=output_path+"return_lr", kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_lr], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename=output_path+"return_lr_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_lr, df=df_test, features=features, h=0.01, x_label=feature_x, y_label=feature_y,
                               vlines=tertile_boundary[feature_x], hlines=tertile_boundary[feature_y], colors=colors,
                               xlim=(-3,3), ylim=(-3,3), figsize=(8,6), ticks=[0,1,2], filename=output_path+"decision_boundary_lr")


    #------------------------------------------
    # Xgboost
    #------------------------------------------
    if run_xgb:
        print("Running xgboost")
        # Set parameters to search
        param_grid = { 'max_depth': [4],
                       'learning_rate': [0.3],
                       'n_estimators': [100],
                       'objective': ['multi:softmax'],
                       'min_child_weight': [1000],
                       'gamma': [10.0], # np.logspace(-2,1,3), # [10]
                       'lambda': [1], # np.logspace(0,2,3)
                       'subsample': [0.5],
                        "n_jobs":[-1],
                       'num_class': [3]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_xgb, cv_results_xgb = grid_search_purged_cv(df_train=df_train,
                                                           model=XGBClassifier(),
                                                           param_grid=param_grid,
                                                           metric=metric,
                                                           features=features, label=label,
                                                           k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_xgb.to_csv(output_path+"cv_results_xgb.csv")
        else:
            #best_params_xgb = {'max_depth': 3, 'learning_rate': 0.042169650342858224, 'n_estimators': 100, 'objective': 'multi:softmax',
            #                   'min_child_weight': 1.0, 'gamma': 10.0, 'lambda': 1, 'subsample': 0.5, 'n_jobs': -1, 'num_class': 3}
            best_params_xgb = {'max_depth': 4, 'learning_rate': 0.3, 'n_estimators': 50, 'objective': 'multi:softmax',
                               'min_child_weight': 1000.0, 'gamma': 10.0, 'lambda': 1, 'subsample': 0.5, 'n_jobs': -1, 'num_class': 3}

        # Calculate cumulative return using best parameters
        df_cum_train_xgb, df_cum_test_xgb, model_xgb = predict_and_calculate_cum_return(model=XGBClassifier(**best_params_xgb),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_xgb, df_cum_test_xgb, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename=output_path+"return_xgb", kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_xgb], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename=output_path+"return_xgb_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_xgb, df=df_test, features=features, h=0.01, x_label=feature_x, y_label=feature_y,
                               vlines=tertile_boundary[feature_x], hlines=tertile_boundary[feature_y], colors=colors,
                               xlim=(-3,3), ylim=(-3,3), figsize=(8,6), ticks=[0,1,2], filename=output_path+"decision_boundary_xgb")




    #------------------------------------------
    # kNN
    #------------------------------------------
    if run_knn:
        print("Running kNN")
        # Set parameters to search
        param_grid = {'n_neighbors': [int(x) for x in np.logspace(1,4,10)]} 
        
        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_knn, cv_results_knn = grid_search_purged_cv(df_train=df_train,
                                                           model=KNeighborsClassifier(),
                                                           param_grid=param_grid,
                                                           metric=metric,
                                                           features=features, label=label,
                                                           k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_knn.to_csv(output_path+"cv_results_knn.csv")
        else:
            best_params_knn = {'n_neighbors': 1000}

        # Calculate cumulative return using best parameters
        df_cum_train_knn, df_cum_test_knn, model_knn = predict_and_calculate_cum_return(model=KNeighborsClassifier(**best_params_knn),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_knn, df_cum_test_knn, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename=output_path+"return_knn", kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_knn], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename=output_path+"return_knn_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_knn, df=df_test, features=features, h=0.01, x_label=feature_x, y_label=feature_y,
                               vlines=tertile_boundary[feature_x], hlines=tertile_boundary[feature_y], colors=colors,
                               xlim=(-3,3), ylim=(-3,3), figsize=(8,6), ticks=[0,1,2], filename=output_path+"decision_boundary_knn")
        
        
    #------------------------------------------
    # Neural network
    #------------------------------------------
    if run_nn:
        print("Running neural network")
        # Set parameters to search
        param_grid = {'activation':['relu'],
                      'hidden_layer_sizes':[(300,300,300,300,300), (500, 400, 300, 200, 100), (1000, 500,400,300,200,100,50,10)],
                      'alpha': np.logspace(-7,-2,5),
                      'early_stopping':[True],
                      'max_iter':[200],
                      'learning_rate':['adaptive'] #['constant', 'adaptive'],
                      }
        
        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_nn, cv_results_nn = grid_search_purged_cv(df_train=df_train,
                                                           model=MLPClassifier(),
                                                           param_grid=param_grid,
                                                           metric=metric,
                                                           features=features, label=label,
                                                           k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_nn.to_csv(output_path+"cv_results_nn.csv")
        else:
            best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (100, 100, 100), 'alpha': 0.046415888336127725, 'early_stopping': True, 'learning_rate': 'adaptive'}
            #best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (500, 400, 300, 200, 100), 'alpha': 4.641588833612782e-06, 'early_stopping': True, 'learning_rate': 'adaptive'}
            # Best performing
            #best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (1000, 500, 400, 300, 200, 100, 50, 10), 'alpha': 7.847599703514606e-05, 'early_stopping': True, 'max_iter': 500, 'learning_rate': 'adaptive'}

        # Calculate cumulative return using best parameters
        df_cum_train_nn, df_cum_test_nn, model_nn = predict_and_calculate_cum_return(model=MLPClassifier(**best_params_nn),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_nn, df_cum_test_nn, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename=output_path+"return_nn", kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_nn], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename=output_path+"return_nn_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_nn, df=df_test, features=features, h=0.01, x_label=feature_x, y_label=feature_y,
                               vlines=tertile_boundary[feature_x], hlines=tertile_boundary[feature_y], colors=colors,
                               xlim=(-3,3), ylim=(-3,3), figsize=(8,6), ticks=[0,1,2], filename=output_path+"decision_boundary_nn")

        
    #------------------------------------------
    # Classification by simple sort
    #------------------------------------------
    if run_sort:
        print("Running sort by %s" % feature_x)
        # Calculate cumulative return
        df_cum_train_sort_x = cumulative_return(df=df_train, var_classes='%s_tertile' % feature_x, total_return=label_fm)
        df_cum_test_sort_x = cumulative_return(df=df_test, var_classes='%s_tertile' % feature_x, total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_sort_x = df_cum_train_sort_x.rename({'%s_tertile' % feature_x:"pred"}, axis=1)
        df_cum_test_sort_x = df_cum_test_sort_x.rename({'%s_tertile' % feature_x:"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_sort_x, df_cum_test_sort_x, label_reg, figsize=(8,6),
                               group_label={2: '%s low' % feature_x, 1: '%s mid' % feature_x, 0: '%s high' % feature_x},
                               kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,5)}, filename=output_path+"return_sort")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_sort_x], list_labels=["Sort %s" % feature_x], label_reg=label_reg,
                                    kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)},
                                    return_label=[0,1,2], # For special case of AG, change it to [2,1,0] so that it calculates Low AG - High AG.
                                    figsize=(8,6), filename=output_path+"return_sort_diff")

        print("Running sort by %s" %feature_y)
        # Calculate cumulative return
        df_cum_train_sort_y = cumulative_return(df=df_train, var_classes='%s_tertile' % feature_y, total_return=label_fm)
        df_cum_test_sort_y = cumulative_return(df=df_test, var_classes='%s_tertile' % feature_y, total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_sort_y = df_cum_train_sort_y.rename({'%s_tertile' % feature_y:"pred"}, axis=1)
        df_cum_test_sort_y = df_cum_test_sort_y.rename({'%s_tertile' % feature_y:"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_sort_y, df_cum_test_sort_y, label_reg, figsize=(8,6),
                               group_label={2: '%s low' % feature_y, 1: '%s mid' % feature_y, 0: '%s high' % feature_y},
                               kwargs_train={'ylim':(-1,7)}, kwargs_test={'ylim':(-1,5)}, filename=output_path+"return_sort_%s" % feature_y)
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_sort_y], list_labels=["Sort %s" % feature_y], label_reg=label_reg,
                                    kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)}, return_label=[0,1,2],
                                    figsize=(8,6), filename=output_path+"return_sort_diff_%s" % feature_y)




    #------------------------------------------
    # Summary plots
    #------------------------------------------
    if run_summary:
        # Read Tensorflow results
        df_cum_test_tf = pickle.load(open(output_path+'df_cum_test_tf.pickle', 'rb'))


        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_lr, df_cum_test_knn, df_cum_test_xgb, df_cum_test_tf],
                                    list_labels=["Linear", "KNN", "XGB", "NN"],
                                    label_reg=label_reg,
                                    figsize=(8,6), return_label=[0,1,2],
                                    kwargs_train={'ylim':(-1,3)},
                                    kwargs_test={'ylim':(-1,3)},
                                    legend_order=["NN", "XGB", "Linear", "KNN"],
                                    filename=output_path+"return_diff_summary")

        # Save results
        save_summary(df_train, label, metric, output_path,
                     cv_results = {'Logistic': cv_results_lr,
                                   'XGB': cv_results_xgb,
                                   'KNN': cv_results_knn,
                                   })



    print("Successfully completed all tasks")


