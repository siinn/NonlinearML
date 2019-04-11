#!/usr/bin/env python
# import common python libraries
from datetime import datetime
import dateutil.relativedelta
import itertools
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from _backtest import *
from _plots import *
from _ML import *
from _HeuristicModel import *

#----------------------------------------------
# Set user options
#----------------------------------------------
# Set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.pd.r4.csv'

# Set features and label
#features = ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'GS', 'SEV', 'CVROIC', 'FCFA']
features = ['FCFA', 'AG']
categories = ['GICSSubIndustryNumber']    
#label_cla = "fqTotalReturn_quintile" # or "fmTotalReturn_quintile"
label_cla = "fqTotalReturn_tertile"
label_reg = "fqTotalReturn" # or "fmTotalReturn"
label_fm = "fmTotalReturn"
time = "eom"


# Set algorithms to run
run_lr              = True
run_xgb             = True
run_knn             = True
run_nn              = True
run_sort_ag         = True
run_ag_fcfa         = True
run_grid_search     = False
run_summary         = False


# Color scheme for plot
#colors = ["#296FAF", "#F3F2F2", "#CC4125"]
colors = ["#2A71B2", "#F3F2F2", "#BA2832"]

#----------------------------------------------
# Functions
#----------------------------------------------
def plot_cumulative_return(df_cum_train, df_cum_test, label_reg, filename, figsize=(15,6), group_label={0:"Q1", 1:"Q2", 2:"Q3"}, time="eom", **kwargs):
    ''' Wrapper of plotting functions. Create cumulative return plot for train and test dataset.
    Args:
        df_cum_train: cumulative return obtained from train set
        df_cum_test: cumulative return obtained from test set
        label_reg: name of target label
        group_label: dictionary to map between label and recognizable string
        time: time column
        filename: filename
    Return:
        None
    ''' 
    print("Plotting cumulative return plots with filename: %s" %filename)

    # plot train dataset
    plot_line_groupby(df=df_cum_train,\
                      x=time, y="cumulative_return",\
                      groupby="pred", group_label = {key:group_label[key]+" (Train)" for key in group_label},\
                      x_label="Time", y_label="Cumulative %s" %label_reg, ylog=False, figsize=figsize, filename = "%s_train" %filename, **kwargs)

    # plot test dataset
    plot_line_groupby(df=df_cum_test,\
                      x=time, y="cumulative_return",\
                      groupby="pred", group_label = {key:group_label[key]+" (Test)" for key in group_label},\
                      x_label="Time", y_label="Cumulative %s" %label_reg, ylog=False, figsize=figsize, filename = "%s_test" %filename, **kwargs)
    return

def plot_cumulative_return_diff(list_cum_returns, list_labels, figsize=(15,6), filename="", **kwargs):
    ''' Wrapper for plotting function. This function plots difference in cumulative return for given models where
        difference in return is defined as Q1+Q2 - Q3.
    Args:
        list_cum_return: list of dataframe representing cumulative returns (output of "predict_and_calculate_cum_return")
        list_label: list of labels for the models 
    '''
    # Calculate difference in return and concatenate
    df_diff_q1q2_q3 = pd.concat([calculate_diff_return(cum_return, output_col=label)[0] for cum_return, label in zip(list_cum_returns, list_labels)])
    df_diff_q1_q3 = pd.concat([calculate_diff_return(cum_return, output_col=label)[1] for cum_return, label in zip(list_cum_returns, list_labels)])

    # plot test dataset
    plot_line_groupby(df=df_diff_q1q2_q3.sort_index(),\
                      x="index", y="cumulative_return",\
                      groupby="pred", group_label = {key:key for key in df_diff_q1q2_q3["pred"].unique()},\
                      x_label="Time", y_label="Cumulative %s\n(Q1+Q2) - Q3" %label_reg, ylog=False, figsize=figsize, filename = "%s_q1q2_q3" %filename, **kwargs)
    plot_line_groupby(df=df_diff_q1_q3.sort_index(),\
                      x="index", y="cumulative_return",\
                      groupby="pred", group_label = {key:key for key in df_diff_q1q2_q3["pred"].unique()},\
                      x_label="Time", y_label="Cumulative %s\nQ1 - Q3" %label_reg, ylog=False, figsize=figsize, filename = "%s_q1_q3" %filename, **kwargs)
    return        


if __name__ == "__main__":

    #------------------------------------------
    # Read dataset
    #------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=None, parse_dates=[time])

    # Assign AG and FCFA tertiles
    df = discretize_variables_by_month(df=df, variables=['AG'], month="eom", labels_tertile=[0,1,2], labels_quintile=[4,3,2,1,0])
    df = discretize_variables_by_month(df=df, variables=['FCFA'], month="eom", labels_tertile=[2,1,0], labels_quintile=[4,3,2,1,0])
    tertile_boundary =  get_tertile_boundary(df, ["AG", "FCFA"])

    # Create train, validation, test dataset
    df_train_all, df_test = train_test_split(df=df, date_column=time, train_length = 140, 
                                             train_end = to_datetime("2006-12-31"),
                                             test_begin = to_datetime("2007-01-01"),
                                             test_end = to_datetime("2017-11-01"))

    df_train, df_val = train_val_split_by_col(df_train_all, col="SecurityID", train_size=0.7)

    #------------------------------------------
    # Print job summary
    #------------------------------------------
    print("Algorithms to run:")
    print(" > run_lr          = %s" % run_lr)
    print(" > run_xgb         = %s" % run_xgb)
    print(" > run_knn         = %s" % run_knn)
    print(" > run_nn          = %s" % run_nn)
    print(" > run_sort_ag     = %s" % run_sort_ag)
    print(" > run_grid_search = %s" % run_grid_search)


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
                  "C": np.logspace(0,2,5)} #[1,100]}
    
        # Perform grid search using validation set
        best_params_lr, df_params_lr = grid_search(model=LogisticRegression(),
                                                   param_grid=param_grid,
                                                   df_train=df_train, df_val=df_val,
                                                   features=features, label_cla=label_cla, label_fm=label_fm)
    
    
        # Calculate cumulative return using best parameters
        df_cum_train_lr, df_cum_test_lr, model_lr = predict_and_calculate_cum_return(model=LogisticRegression(**best_params_lr),
                                                                 df_train=df_train_all, df_test=df_test,
                                                                 features=features, label_cla=label_cla, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_lr, df_cum_test_lr, label_reg, figsize=(8,6), filename="cum_lr", ylim=(-0.6, 3.0))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_lr], list_labels=["OLS (ridge)"], figsize=(8,6), ylim=(-0.6,5), filename="cum_lr_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_lr, df=df_test, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="lr")


    #------------------------------------------
    # Xgboost
    #------------------------------------------
    if run_xgb:
        print("Running xgboost")
        # Set parameters to search
        param_grid = { 'max_depth': [3, 5],
                       'learning_rate': np.logspace(-4, -0.5, 5),
                       'n_estimators': [100],
                       'objective': ['multi:softmax'],
                       'min_child_weight': np.logspace(0,2,5),
                       'gamma': np.logspace(0,3,5),
                       'lambda': [1], #np.logspace(0,3,5),
                       'subsample': [0.4, 0.6],
                        "n_jobs":[-1],
                       'num_class': [3]}


        # Perform grid search using validation set
        if run_grid_search:
            best_params_xgb, df_params_xgb = grid_search(model=XGBClassifier(),
                                                         param_grid=param_grid,
                                                         df_train=df_train, df_val=df_val,
                                                         features=features, label_cla=label_cla, label_fm=label_fm)
        else:
            # Also good
            best_params_xgb = {'max_depth': 3, 'learning_rate': 0.0007498942093324559, 'n_estimators': 100, 'objective': 'multi:softmax',
                               'min_child_weight': 10.0, 'gamma': 1.0, 'lambda': 1.0, 'subsample': 0.6, 'n_jobs': -1, 'num_class': 3}
            # Overfitting
            #best_params_xgb = {'max_depth': 10, 'learning_rate': 0.0007498942093324559, 'n_estimators': 50, 'objective': 'multi:softmax',
            #                   'min_child_weight': 10.0, 'gamma': 1.0, 'lambda': 1.0, 'subsample': 1, 'n_jobs': -1, 'num_class': 3}

        # Calculate cumulative return using best parameters
        df_cum_train_xgb, df_cum_test_xgb, model_xgb = predict_and_calculate_cum_return(model=XGBClassifier(**best_params_xgb),
                                                                             df_train=df_train_all, df_test=df_test,
                                                                             features=features, label_cla=label_cla, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_xgb, df_cum_test_xgb, label_reg, figsize=(8,6), filename="cum_xgb", ylim=(-0.6, 3.0))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_xgb], list_labels=["XGBoost"], figsize=(8,6), ylim=(-0.6,5), filename="cum_xgb_diff")


        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_xgb, df=df_test, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="xgb")


    #------------------------------------------
    # kNN
    #------------------------------------------
    if run_knn:
        print("Running kNN")
        # Set parameters to search
        param_grid = {'n_neighbors': [int(x) for x in np.logspace(1,3,30)]} 
        
        # Perform grid search using validation set
        if run_grid_search:
            best_params_knn, df_params_knn = grid_search(model=KNeighborsClassifier(),
                                                         param_grid=param_grid,
                                                         df_train=df_train, df_val=df_val,
                                                         features=features, label_cla=label_cla, label_fm=label_fm)
        else:
            best_params_knn = {'n_neighbors': 500} 
            # Overfitting
            #best_params_knn = {'n_neighbors': 10} 
        
        # Calculate cumulative return using best parameters
        df_cum_train_knn, df_cum_test_knn, model_knn = predict_and_calculate_cum_return(model=KNeighborsClassifier(**best_params_knn),
                                                                             df_train=df_train_all, df_test=df_test,
                                                                             features=features, label_cla=label_cla, label_fm=label_fm)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_knn, df_cum_test_knn, label_reg, figsize=(8,6), filename="cum_knn", ylim=(-0.6, 3.0))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_knn], list_labels=["kNN"], figsize=(8,6), ylim=(-0.6,5), filename="cum_knn_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_knn, df=df_test, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
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
        
        # Perform grid search using validation set
        if run_grid_search:
            best_params_nn, df_params_nn = grid_search(model=MLPClassifier(),
                                                         param_grid=param_grid,
                                                         df_train=df_train, df_val=df_val,
                                                         features=features, label_cla=label_cla, label_fm=label_fm)
        else:
            #best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (100, 100, 100), 'alpha': 0.046415888336127725, 'early_stopping': True, 'learning_rate': 'adaptive'}
            #best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (500, 400, 300, 200, 100), 'alpha': 4.641588833612782e-06, 'early_stopping': True, 'learning_rate': 'adaptive'}

            #best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (1000, 500, 400, 300, 200, 100, 50, 10), 'alpha': 0.0004832930238571752, 'early_stopping': False, 'max_iter': 500, 'learning_rate': 'adaptive'}
            best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (1000, 500, 400, 300, 200, 100, 50, 10), 'alpha': 7.847599703514606e-05, 'early_stopping': True, 'max_iter': 500, 'learning_rate': 'adaptive'}

        # Calculate cumulative return using best parameters
        df_cum_train_nn, df_cum_test_nn, model_nn = predict_and_calculate_cum_return(model=MLPClassifier(**best_params_nn),
                                                                             df_train=df_train_all, df_test=df_test,
                                                                             features=features, label_cla=label_cla, label_fm=label_fm)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_nn, df_cum_test_nn, label_reg, figsize=(8,6), filename="cum_nn", ylim=(-0.6, 3.0))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_nn], list_labels=["Neural Net"], figsize=(8,6), ylim=(-0.6,5), filename="cum_nn_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=model_nn, df=df_test, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="nn")

        
        
    #------------------------------------------
    # Classification by simple sort (AG)
    #------------------------------------------
    if run_sort_ag:
        print("Running sort by AG")
        # Calculate cumulative return
        df_cum_train_simple = cumulative_return(df=df_train_all, var_classes='AG_tertile', total_return=label_fm)
        df_cum_test_simple = cumulative_return(df=df_test, var_classes='AG_tertile', total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_simple = df_cum_train_simple.rename({"AG_tertile":"pred"}, axis=1)
        df_cum_test_simple = df_cum_test_simple.rename({"AG_tertile":"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_simple, df_cum_test_simple, label_reg, figsize=(8,6), group_label={2: 'AG low', 1: 'AG mid', 0: 'AG high'}, filename="cum_simple")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple], list_labels=["Sort AG"], figsize=(8,6), ylim=(-0.6,5), filename="cum_simple_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=HeuristicModel_SortAG(df=df, feature="AG"),
                               df=df_train_all, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="sort_ag")


    #------------------------------------------
    # Classification by simple AG and FCFA
    #------------------------------------------
    if run_ag_fcfa:
        print("Running heuristic model using AG and high FCFA")
        # Calculate cumulative return
        df_cum_train_simple_ag_fcfa = cumulative_return(df=HeuristicModel_AG_HighFCFA(df=df_train_all, ag="AG", fcfa="FCFA")\
                                                   .predict_exact(df_train_all),
                                                var_classes='pred', total_return=label_fm)
        df_cum_test_simple_ag_fcfa = cumulative_return(df=HeuristicModel_AG_HighFCFA(df=df_test, ag="AG", fcfa="FCFA")\
                                                   .predict_exact(df_test),
                                                var_classes='pred', total_return=label_fm)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_simple_ag_fcfa, df_cum_test_simple_ag_fcfa, label_reg, figsize=(8,6), group_label={2: 'Low', 1: 'Mid', 0: 'High'}, filename="cum_simple_ag_fcfa")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple_ag_fcfa], list_labels=["Heuristic"], figsize=(8,6), ylim=(-0.6,5), filename="cum_simple_ag_fcfa_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=HeuristicModel_AG_HighFCFA(df=df, ag="AG", fcfa="FCFA"),
                               df=df_train_all, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="heuristics_ag_fcfa")
       

    #------------------------------------------
    # Summary plots
    #------------------------------------------
    if run_summary:
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple, df_cum_test_simple_ag_fcfa, df_cum_test_lr,
                                                      df_cum_test_knn, df_cum_test_xgb, df_cum_test_nn],
                                    list_labels=["Sort AG", "Heuristic", "Linear", "KNN", "XGB", "NN"],
                                    legend_order=["XGB", "Linear", "NN", "KNN", "Heuristic", "Sort AG"],
                                    figsize=(8,6),
                                    filename="diff_return_test")

        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple, df_cum_test_simple_ag_fcfa, df_cum_test_lr,
                                                      df_cum_test_knn, df_cum_test_xgb],
                                    list_labels=["Sort AG", "Heuristic", "Linear", "KNN", "XGB"],
                                    #legend_order=["XGB", "Linear", "KNN", "Heuristic", "Sort AG"],
                                    legend_order=["XGB", "Linear", "KNN", "Sort AG", "Heuristic"],
                                    figsize=(8,6),
                                    filename="diff_return_test")



        plot_cumulative_return_diff(list_cum_returns=[
                                                      df_cum_test_simple_ag_fcfa,
                                                      df_cum_test_simple,
                                                     ],
                                    list_labels=[
                                                "Heuristic",
                                                "Sort AG",
                                                ],
                                    figsize=(8,6),
                                    ylim=(-0.6,5),
                                    filename="diff_return_test_simple")





    '''========================================================='''
    '''

    To-do:
    
        1 finish optimization on NN
        2 create overfit plot
        3 What's next?
            - intuition -> ML
            - intuition <- ML

    '''
    '''========================================================='''







    print("Successfully completed all tasks")


