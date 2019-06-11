#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import numpy as np
import pandas as pd
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
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.p1.csv'

# Set available features and labels
#features = ['GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC', 'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum']
features = ['AG', 'FCFA']

# Set labels
label = "fqTotalReturn_tertile"     # or "fmTotalReturn_quintile"
label_reg = "fqTotalReturn"         # or "fmTotalReturn"
label_fm = "fmTotalReturn"          # Used to calculate cumulative return
date_column = "eom"

# Set train and test period
test_begin = "2011-01-01"
test_end = "2017-11-01"

# k-fold
k = 10

# Set color scheme for decision boundary plot
colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]

# Set algorithms to run
run_lr              = True
run_xgb             = True
run_knn             = True
run_nn              = True
run_sort_ag         = True
run_grid_search     = False
run_summary         = True

#------------------------------------------
# Main 
#------------------------------------------
if __name__ == "__main__":

    #------------------------------------------
    # Read dataset
    #------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=[0], parse_dates=[date_column])

    # Assign AG and FCFA tertiles
    df = discretize_variables_by_month(df=df, variables=['AG', 'FCFA'], month="eom",
                                       labels_tertile={'AG':[2,1,0], 'FCFA':[2,1,0]}, # 0 is high
                                       labels_quintile={'AG':[4,3,2,1,0], 'FCFA':[4,3,2,1,0]}) # 0 is high
    tertile_boundary =  get_tertile_boundary(df, ["AG", "FCFA"])

    # Split dataset into train and test dataset
    df_train, df_test = train_test_split_by_date(df, date_column, test_begin, test_end)


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
                  "C": np.logspace(0,2,1)} #[1,100]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_lr, cv_results = grid_search_purged_cv(df_train=df_train,
                                                           model=LogisticRegression(),
                                                           param_grid=param_grid,
                                                           metric='accuracy',
                                                           features=features, label=label,
                                                           k=3, purge_length=3, verbose=True)
        else:
            best_params_lr = {'penalty': 'l2', 'multi_class': 'multinomial', 'solver': 'newton-cg', 'max_iter': 100, 'n_jobs': -1, 'C': 1.0}

        # Calculate cumulative return using best parameters
        df_cum_train_lr, df_cum_test_lr, model_lr = predict_and_calculate_cum_return(model=LogisticRegression(**best_params_lr),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_lr, df_cum_test_lr, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename="cum_lr", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_lr], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename="cum_lr_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_lr, df=df_test, features=features, h=0.01, x_label="AG", y_label="FCFA",
                               vlines=tertile_boundary["AG"], hlines=tertile_boundary["FCFA"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="lr")


    #------------------------------------------
    # Xgboost
    #------------------------------------------
    if run_xgb:
        print("Running xgboost")
        # Set parameters to search
        param_grid = { 'max_depth': [3],
                       'learning_rate': np.logspace(-4, -0.5, 1),
                       'n_estimators': [100],
                       'objective': ['multi:softmax'],
                       'min_child_weight': np.logspace(0,2,1),
                       'gamma': np.logspace(0,3,1),
                       'lambda': [1], #np.logspace(0,3,5),
                       'subsample': [0.4],
                        "n_jobs":[-1],
                       'num_class': [3]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_xgb, cv_results = grid_search_purged_cv(df_train=df_train,
                                                           model=XGBClassifier(),
                                                           param_grid=param_grid,
                                                           metric='accuracy',
                                                           features=features, label=label,
                                                           k=3, purge_length=3, verbose=True)
        else:
            best_params_xgb = {'max_depth': 3, 'learning_rate': 0.0007498942093324559, 'n_estimators': 100, 'objective': 'multi:softmax',
                               'min_child_weight': 10.0, 'gamma': 1.0, 'lambda': 1.0, 'subsample': 0.6, 'n_jobs': -1, 'num_class': 3}

        # Calculate cumulative return using best parameters
        df_cum_train_xgb, df_cum_test_xgb, model_xgb = predict_and_calculate_cum_return(model=XGBClassifier(**best_params_xgb),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_xgb, df_cum_test_xgb, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename="cum_xgb", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_xgb], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename="cum_xgb_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_xgb, df=df_test, features=features, h=0.01, x_label="AG", y_label="FCFA",
                               vlines=tertile_boundary["AG"], hlines=tertile_boundary["FCFA"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="xgb")




    #------------------------------------------
    # kNN
    #------------------------------------------
    if run_knn:
        print("Running kNN")
        # Set parameters to search
        param_grid = {'n_neighbors': [int(x) for x in np.logspace(1,3,10)]} 
        
        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_knn, cv_results = grid_search_purged_cv(df_train=df_train,
                                                           model=KNeighborsClassifier(),
                                                           param_grid=param_grid,
                                                           metric='accuracy',
                                                           features=features, label=label,
                                                           k=3, purge_length=3, verbose=True)
        else:
            best_params_knn = {'n_neighbors': 500}

        # Calculate cumulative return using best parameters
        df_cum_train_knn, df_cum_test_knn, model_knn = predict_and_calculate_cum_return(model=KNeighborsClassifier(**best_params_knn),
                                                                 df_train=df_train, df_test=df_test,
                                                                 features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_knn, df_cum_test_knn, label_reg, group_label={0:'T1', 1:'T2', 2:'T3'},
                               figsize=(8,6), filename="cum_knn", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_knn], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename="cum_knn_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_knn, df=df_test, features=features, h=0.01, x_label="AG", y_label="FCFA",
                               vlines=tertile_boundary["AG"], hlines=tertile_boundary["FCFA"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="knn")
        
        
    #------------------------------------------
    # Neural network
    #------------------------------------------
    if run_nn:
        print("Running neural network")
        # Set parameters to search
        param_grid = {'activation':['relu'],
                      'hidden_layer_sizes':[(500, 400, 300, 200, 100), (1000, 500,400,300,200,100,50,10)],
                      'alpha': np.logspace(-7,-2,20),
                      'early_stopping':[True, False],
                      'max_iter':[200,500],
                      'learning_rate':['adaptive'] #['constant', 'adaptive'],
                      }
        
        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_nn, cv_results = grid_search_purged_cv(df_train=df_train,
                                                           model=MLPClassifier(),
                                                           param_grid=param_grid,
                                                           metric='accuracy',
                                                           features=features, label=label,
                                                           k=3, purge_length=3, verbose=True)
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
                               figsize=(8,6), filename="cum_nn", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_nn], return_label=[0,1,2],
                                    list_labels=["Logistic regression"], label_reg=label_reg, figsize=(8,6),
                                    filename="cum_nn_diff", kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_nn, df=df_test, features=features, h=0.01, x_label="AG", y_label="FCFA",
                               vlines=tertile_boundary["AG"], hlines=tertile_boundary["FCFA"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="nn")

        
    #------------------------------------------
    # Classification by simple sort (AG)
    #------------------------------------------
    if run_sort_ag:
        print("Running sort by AG")
        # Calculate cumulative return
        df_cum_train_simple = cumulative_return(df=df_train, var_classes='AG_tertile', total_return=label_fm)
        df_cum_test_simple = cumulative_return(df=df_test, var_classes='AG_tertile', total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_simple = df_cum_train_simple.rename({"AG_tertile":"pred"}, axis=1)
        df_cum_test_simple = df_cum_test_simple.rename({"AG_tertile":"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_simple, df_cum_test_simple, label_reg, figsize=(8,6),
                               group_label={2: 'AG low', 1: 'AG mid', 0: 'AG high'},
                               kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)}, filename="cum_simple")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple], list_labels=["Sort AG"], label_reg=label_reg,
                                    kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)},
                                    return_label=[2,1,0], # THIS IS A SPECIAL CASE. It will calculate Low AG - High AG.
                                    figsize=(8,6), filename="cum_simple_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=HeuristicModel_SortAG(df=df, feature="AG"),
                               df=df_train, features=features, h=0.01, x_label="AG", y_label="FCFA",
                               vlines=tertile_boundary["AG"], hlines=tertile_boundary["FCFA"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="sort_ag")

        print("Running sort by FCFA")
        # Calculate cumulative return
        df_cum_train_simple_FCFA = cumulative_return(df=df_train, var_classes='FCFA_tertile', total_return=label_fm)
        df_cum_test_simple_FCFA = cumulative_return(df=df_test, var_classes='FCFA_tertile', total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_simple_FCFA = df_cum_train_simple_FCFA.rename({"FCFA_tertile":"pred"}, axis=1)
        df_cum_test_simple_FCFA = df_cum_test_simple_FCFA.rename({"FCFA_tertile":"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_simple_FCFA, df_cum_test_simple_FCFA, label_reg, figsize=(8,6),
                               group_label={2: 'FCFA low', 1: 'FCFA mid', 0: 'FCFA high'},
                               kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)}, filename="cum_simple_FCFA")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple_FCFA], list_labels=["Sort FCFA"], label_reg=label_reg,
                                    kwargs_train={'ylim':(-1,5)}, kwargs_test={'ylim':(-1,3)}, return_label=[0,1,2],
                                    figsize=(8,6), filename="cum_simple_diff_FCFA")




    #------------------------------------------
    # Summary plots
    #------------------------------------------
    if run_summary:
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_lr, df_cum_test_knn, df_cum_test_xgb, df_cum_test_nn],
                                    list_labels=["Linear", "KNN", "XGB", "NN"],
                                    label_reg=label_reg,
                                    figsize=(8,6), return_label=[0,1,2],
                                    kwargs_train={'ylim':(-1,5)},
                                    kwargs_test={'ylim':(-1,3)},
                                    legend_order=["NN", "XGB", "Linear", "KNN"],
                                    filename="diff_return_test")








#
#
#
#
#
#
#
