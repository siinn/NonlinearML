#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

# Import custom libraries
from Asset_growth.lib.backtest import *
from Asset_growth.lib.plots import *
from Asset_growth.lib.utils import *
from Asset_growth.lib.heuristicModel import *

#----------------------------------------------
# Set user options
#----------------------------------------------
# Set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.pd.r4.csv'

# Set features
''' Set features of interest. This algorithm takes a pair of features. The order of pair matters.
    ex. ['FCFA', 'AG'] will generate decision boundary with x=FCFA, y=AG.
    Full list of available features:
        ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'GS', 'SEV', 'CVROIC', 'FCFA']
'''
features = ['FCFA', 'AG']
categories = ['GICSSubIndustryNumber']    

# Set labels
label_cla = "fqTotalReturn_tertile"     # or "fmTotalReturn_quintile"
label_reg = "fqTotalReturn"             # or "fmTotalReturn"
label_fm = "fmTotalReturn"              # Used to calculate cumulative return
time = "eom"

# Set train and test period
train_length = 120
train_end = "2006-12-31"
test_begin = "2007-01-01"
test_end = "2017-11-01"

# Train/validation split ratio
train_validation_split = 0.7

# Set algorithms to run
run_lr              = True
run_xgb             = True
run_knn             = True
run_nn              = True
run_sort_ag         = True
run_ag_fcfa         = True
run_grid_search     = False
run_summary         = True

# Color scheme for plot
# green, gray, red
colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"]







#------------------------------------------
# Main 
#------------------------------------------
if __name__ == "__main__":

    #------------------------------------------
    # Read dataset
    #------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=None, parse_dates=[time])

    # Assign AG and FCFA tertiles
    df = discretize_variables_by_month(df=df, variables=['AG', 'FCFA'], month="eom",
                                       labels_tertile={'AG':[0,1,2], 'FCFA':[2,1,0]},
                                       labels_quintile={'AG':[4,3,2,1,0], 'FCFA':[4,3,2,1,0]})
    tertile_boundary =  get_tertile_boundary(df, ["AG", "FCFA"])

    #------------------------------------------
    # Create train, validation, test dataset
    #------------------------------------------
    # Split train and test by dates
    df_train_all, df_test = train_test_split(df=df, date_column=time, train_length = train_length, 
                                             train_end = to_datetime(train_end),
                                             test_begin = to_datetime(test_begin),
                                             test_end = to_datetime(test_end))

    # Split train into train and validation datasets by SecurityID (cross-sectional)
    df_train, df_val = train_val_split_by_col(df_train_all, col="SecurityID", train_size=train_validation_split)



    #------------------------------------------
    # Print job summary
    #------------------------------------------
    print("Algorithms to run:")
    print(" > run_lr          = %s" % run_lr)
    print(" > run_xgb         = %s" % run_xgb)
    print(" > run_knn         = %s" % run_knn)
    print(" > run_nn          = %s" % run_nn)
    print(" > run_sort_ag     = %s" % run_sort_ag)
    print(" > run_ag_fcfa     = %s" % run_ag_fcfa)
    print(" > run_grid_search = %s" % run_grid_search)
    print(" > run_summary = %s" % run_summary)

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
        plot_cumulative_return(df_cum_train_lr, df_cum_test_lr, label_reg, figsize=(8,6), filename="cum_lr", ylim=(-0.25,2))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_lr], list_labels=["OLS (ridge)"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_lr_diff")

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

        # Calculate cumulative return using best parameters
        df_cum_train_xgb, df_cum_test_xgb, model_xgb = predict_and_calculate_cum_return(model=XGBClassifier(**best_params_xgb),
                                                                             df_train=df_train_all, df_test=df_test,
                                                                             features=features, label_cla=label_cla, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_xgb, df_cum_test_xgb, label_reg, figsize=(8,6), filename="cum_xgb", ylim=(-0.25,2))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_xgb], list_labels=["XGBoost"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_xgb_diff")


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
        param_grid = {'n_neighbors': [int(x) for x in np.logspace(1,3,10)]} 
        
        # Perform grid search using validation set
        # Entire training set is used for grid search as sample size is crucial in kNN.
        if run_grid_search:
            best_params_knn, df_params_knn = grid_search(model=KNeighborsClassifier(),
                                                         param_grid=param_grid,
                                                         df_train=df_train_all, df_val=df_train_all,
                                                         features=features, label_cla=label_cla, label_fm=label_fm)
        else:
            best_params_knn = {'n_neighbors': 500} 
        
        # Calculate cumulative return using best parameters
        df_cum_train_knn, df_cum_test_knn, model_knn = predict_and_calculate_cum_return(model=KNeighborsClassifier(**best_params_knn),
                                                                             df_train=df_train_all, df_test=df_test,
                                                                             features=features, label_cla=label_cla, label_fm=label_fm)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_knn, df_cum_test_knn, label_reg, figsize=(8,6), filename="cum_knn", ylim=(-0.25,2))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_knn], list_labels=["kNN"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_knn_diff")

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
            # Best performing
            best_params_nn = {'activation': 'relu', 'hidden_layer_sizes': (1000, 500, 400, 300, 200, 100, 50, 10), 'alpha': 7.847599703514606e-05, 'early_stopping': True, 'max_iter': 500, 'learning_rate': 'adaptive'}

        # Calculate cumulative return using best parameters
        df_cum_train_nn, df_cum_test_nn, model_nn = predict_and_calculate_cum_return(model=MLPClassifier(**best_params_nn),
                                                                             df_train=df_train_all, df_test=df_test,
                                                                             features=features, label_cla=label_cla, label_fm=label_fm)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_nn, df_cum_test_nn, label_reg, figsize=(8,6), filename="cum_nn", ylim=(-0.25,2))
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_nn], list_labels=["Neural Net"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_nn_diff")

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
        plot_cumulative_return(df_cum_train_simple, df_cum_test_simple, label_reg, figsize=(8,6), group_label={2: 'AG low', 1: 'AG mid', 0: 'AG high'}, ylim=(-0.25,2), filename="cum_simple")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple], list_labels=["Sort AG"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_simple_diff")

        # Plot decision boundary of trained model
        plot_decision_boundary(model=HeuristicModel_SortAG(df=df, feature="AG"),
                               df=df_train_all, features=features, h=0.01, x_label="FCFA", y_label="AG",
                               vlines=tertile_boundary["FCFA"], hlines=tertile_boundary["AG"], colors=colors,
                               xlim=False, ylim=False, figsize=(8,6), ticks=[0,1,2], filename="sort_ag")

        print("Running sort by FCFA")
        # Calculate cumulative return
        df_cum_train_simple_FCFA = cumulative_return(df=df_train_all, var_classes='FCFA_tertile', total_return=label_fm)
        df_cum_test_simple_FCFA = cumulative_return(df=df_test, var_classes='FCFA_tertile', total_return=label_fm)
    
        # Rename column before concat
        df_cum_train_simple_FCFA = df_cum_train_simple_FCFA.rename({"FCFA_tertile":"pred"}, axis=1)
        df_cum_test_simple_FCFA = df_cum_test_simple_FCFA.rename({"FCFA_tertile":"pred"}, axis=1)
        
        # Make cumulative return plot
        plot_cumulative_return(df_cum_train_simple_FCFA, df_cum_test_simple_FCFA, label_reg, figsize=(8,6), group_label={2: 'FCFA low', 1: 'FCFA mid', 0: 'FCFA high'}, ylim=(-0.25,2), filename="cum_simple_FCFA")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple_FCFA], list_labels=["Sort FCFA"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_simple_diff_FCFA")



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
        plot_cumulative_return(df_cum_train_simple_ag_fcfa, df_cum_test_simple_ag_fcfa, label_reg, figsize=(8,6), group_label={2: 'Low', 1: 'Mid', 0: 'High'}, ylim=(-0.25,2), filename="cum_simple_ag_fcfa")
        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple_ag_fcfa], list_labels=["Heuristic"], label_reg=label_reg, figsize=(8,6), ylim=(-0.25,2), filename="cum_simple_ag_fcfa_diff")

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
                                    legend_order=["XGB", "NN", "Linear", "KNN", "Heuristic", "Sort AG"],
                                    ylim=(-0.25,2),
                                    label_reg=label_reg,
                                    figsize=(8,6),
                                    filename="diff_return_test")

        plot_cumulative_return_diff(list_cum_returns=[df_cum_test_simple, df_cum_test_simple_ag_fcfa, df_cum_test_lr,
                                                      df_cum_test_knn, df_cum_test_xgb],
                                    list_labels=["Sort AG", "Heuristic", "Linear", "KNN", "XGB"],
                                    #legend_order=["XGB", "Linear", "KNN", "Heuristic", "Sort AG"],
                                    legend_order=["XGB", "Linear", "KNN", "Heuristic", "Sort AG"],
                                    ylim=(-1,4),
                                    label_reg=label_reg,
                                    figsize=(8,6),
                                    filename="diff_return_test")


    #------------------------------------------
    # testing interpretation
    #------------------------------------------
    if False:

        from skater.core.explanations import Interpretation
        from skater.model import InMemoryModel

        # feature importance
        interpreter = Interpretation(df_test[features], feature_names=features)
        model = InMemoryModel(model_xgb.predict_proba, examples=df_train_all[features])
        plots = interpreter.feature_importance.plot_feature_importance(model, ascending = False)

       # PDP
        pyint_model = InMemoryModel(model_xgb.predict_proba, examples=df_train_all[features], target_names=[0,1,2])
        axes_list = interpreter.partial_dependence.plot_partial_dependence(['AG'],
                                                                           pyint_model, 
                                                                           grid_resolution=10, 
                                                                           with_variance=True,
                                                                           figsize = (8, 5))

        for i, fig in enumerate(axes_list[0][:int(len(axes_list[0])/2)]):
            #fig, ax = plt.subplots(1,1, figsize=(8,4))
            #ax = axes_list[0][4]
            #ax.set_ylim(0,1)
            fig.tight_layout()
            fig.savefig("plots/pdp_ag_class%s.png" %i)

        from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
        model = InMemoryModel(model_xgb.predict_proba, examples=df_train_all[features], target_names=[0,1,2])
        pdp2d = interpreter.partial_dependence.plot_partial_dependence([('AG', 'FCFA')], model, grid_resolution=10)        

        for i, fig in enumerate(pdp2d[0][:int(len(pdp2d[0])/2)]):
            fig.tight_layout()
            fig.savefig("plots/pdp2d_class%s.png" %i)
        


    print("Successfully completed all tasks")

