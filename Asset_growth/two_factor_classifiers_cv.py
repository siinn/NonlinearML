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
run_lr = False
run_xgb = True
run_knn = False
run_nn = False
run_sort = False
run_summary = False



#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)

#-------------------------------------------------------------------------------
# Define custom function
#-------------------------------------------------------------------------------
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison
def anova_test(df_data, verbose=False):
    """ Perform Anova test.

    Given K groups with n samples from each group, perform f-test and
    return f-stat and the corresponding p-values
    Args:
        data: Pandas dataframe representing data in the following format 
            model_i run_j, ..., run_n
            model_K run_j, ..., run_n
    Returns:
        f_stat, p_values
    """
    # Run ANOVA
    f_stat, p_value = stats.f_oneway(
        *[df_data.iloc[i].values for i in range(df_data.shape[0])])
    # Calculate degrees of freedom
    dof1 = df_data.shape[0] - 1 # K-1 where K is number of models
    dof2 = df_data.shape[1] - df_data.shape[0] # K-N
    # Print results
    if verbose:
        print(" >> F-statistics = %s, p-value = %s" % (f_stat, p_value))
        print(" >> DOF1 (K-1) = %s, DOF2 (N-K) = %s" % (dof1, dof2))
    return round(f_stat, 6), round(p_value, 6)

def post_hoc_test(df_data):
    """ Perform post hoc test (Tukey)
    Args:
        data: Pandas dataframe representing data in the following format
            model_i run_j, ..., run_n
            model_K run_j, ..., run_n
    Returns:
        df_posthoc: results in Pandas dataframe format
    """
    # Stack data from all groups
    df_stacked = df_data.stack().reset_index().rename(
        {'level_0': 'model','level_1': 'run',0: 'perf'}, axis=1)
    # Perform post hoc test
    post_hoc = MultiComparison(df_stacked['perf'], df_stacked['model'])
    df_posthoc = pd.DataFrame(post_hoc.tukeyhsd(alpha=0.05).summary().data)
    return df_posthoc

def select_best_model_by_anova(cv_results, cv_metric, param_grid):
    """ Select best model by performing ANOVA and post-hoc test.
    
    The preferred model is selected among the models within the statistical
    uncertainties of the model with the highest score.
    
    Minimum index is selected because cv_results is sorted by the
    preferred order of hyperparameters and values in param_grid.

    Args:
        cv_results: Output of CV method (grid_search_purged_cv)
        cv_metric: Metric to use in model selction. e.g. 'f1-score'
        param_grid: Hyperparameter grid used to return best hyperparameters
    Returns:
        f_stats: f-statistics of all models calculated using each metric
        p_values: p-values of all models calculated using each metric
        post_hoc_results: results of post-hoc tests 
        best_params: best parameter selected

    """
    def _convert_to_float(x):
        """ Attempts to convert to float if possible."""
        try:
            return float(x)
        except ValueError:
            return x
    # Extract list of metrics
    metric_names = [
        metric[:-7] for metric in cv_results.columns if 'values' in metric]
    metric_values = [
        metric for metric in cv_results.columns if 'values' in metric]
    # Convert string to list
    """ Example: '[0.0, 0.1, 0.5]' to ['0.0', '0.1', '0.5']. """
    if type(cv_results[metric_values].iloc[0,0]) == str:
        cv_results[metric_values] = cv_results[metric_values].applymap(
            lambda x: x.strip('[]').split(', '))
    # Perform ANOVA and post hoc test on each metrics
    f_stats = {}
    p_values = {}
    post_hoc_results = {}
    # Loop over each metric
    for metric in metric_names:
        metric_values = metric+"_values"
        # Extract values of CV results
        model_perf = pd.DataFrame(cv_results[metric_values].tolist())\
                        .astype(float)
        # Perform one way ANOVA
        print("Performing ANOVA on model performance (%s)" % metric)
        f_stats[metric], p_values[metric] = anova_test(model_perf)
        # If p-value is less than the threshold, perform post hoc test.
        if p_values[metric] < p_thres:
            post_hoc_results[metric] = post_hoc_test(model_perf)
    # Index of the model with the highest score
    print("Model %s has the highest score" % id_max)
    id_max = cv_results[cv_metric].idxmax()
    # Check if we passed ANOVA given metric
    if cv_metric in post_hoc_results.keys():
        # Filter result by the model with the highest score
        post_hoc_top = post_hoc_results[cv_metric].loc[
            (post_hoc_results[cv_metric][0]==id_max) |
            (post_hoc_results[cv_metric][1]==id_max)]
        # Check if there are multiple models within statistical uncertainties
        num_candidates = post_hoc_top.loc[post_hoc_top[5]==False].shape[0]
        if num_candidates == True:
            print("There is only one model with highest %s score." % cv_metric)
            id_selected_model = id_max
        else:
            print("There are %s model with highest %s score" 
                % (num_candidates, cv_metric)
                , " within statistical uncertainties.")
            # Select preferred model
            id_selected_model = min(
                post_hoc_top.loc[post_hoc_top[5]==False][0].min(),
                post_hoc_top.loc[post_hoc_top[5]==False][1].min())
            print("Model %s is selected by preference" % id_selected_model)
    else:
        print("ANOVA failed for %s. Model is selected by preferred order"
            % cv_metric)
        id_selected_model = id_max
    # Recreate hyperparameter combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Return parameters of the preferred model
    best_params = experiments[id_selected_model]
    return {
        "f_stats": f_stats,
        "p_values": p_values,
        "tukey_all_results": post_hoc_results, 
        "tukey_top_results": post_hoc_top,
        "best_params": best_params}



            


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
            "C": np.logspace(-5, -1, 5)} #[1, 100]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_lr, cv_results_lr = grid_search_purged_cv(
                df_train=df_train,
                model=LogisticRegression(),
                param_grid=param_grid,
                metric=metric,
                features=features, label=label,
                k=k, purge_length=3, verbose=False,)
            # Save cross-validation results
            cv_results_lr.to_csv(output_path+"cv_results_lr.csv")
        else:
            best_params_lr = {
                'penalty': 'l2', 'multi_class': 'multinomial',
                'solver': 'newton-cg', 'max_iter': 100, 'n_jobs': -1,
                'C': 0.001}

        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_lr, filename=output_path+"cv_results_lr",
            n_bins=10, figsize=(8, 5), alpha=0.6, hist_type='step')

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
            figsize=(8, 6), filename=output_path+"return_lr",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_lr], return_label=[0, 1, 2],
            list_labels=["Logistic regression"],
            label_reg=label_reg, figsize=(8, 6),
            filename=output_path+"return_lr_diff",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(
            model=model_lr, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            filename=output_path+"decision_boundary_lr")


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
            'n_estimators': [50],
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
                k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_xgb.to_csv(output_path+"cv_results_xgb.csv")
            # Perform ANOVA to select best model
            anova_results_xgb = select_best_model_by_anova(
                cv_results_xgb, cv_metric, param_grid)
        else:
            best_params_xgb = {
                'max_depth': 5, 'learning_rate': 0.3, 'n_estimators': 50,
                'objective': 'multi:softmax', 'min_child_weight': 1000.0,
                'gamma': 10.0, 'lambda': 1, 'subsample': 0.5, 'n_jobs': -1,
                'num_class': 3}


        #-----------------------------------------------------------------------
        # Cross-validation sensitivity study
        #-----------------------------------------------------------------------
        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_xgb, filename=output_path+"cv_results_xgb_hist",
            n_bins=10, x_range=None,
            legend_loc=None, legend_box=(1, 1), figsize=(18, 10), alpha=0.7,
            hist_type='stepfilled', edgecolor='black')
        plot_cv_box(
            cv_results_xgb, filename=output_path+"xgb/cv_results_xgb_box_%s" %cv_metric,
            cv_metric=cv_metric, figsize=(18, 10), color="#3399FF")
            #palette=sns.color_palette("GnBu", n_colors=15))

        # Plot decision boundary plots of all hyperparameter sets
        plot_decision_boundary_multiple_hparmas(
            param_grid=param_grid, label=label, model=XGBClassifier(),
            df=df_train, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y],
            colors=colors, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8),
            ticks=[0, 1, 2], filename=output_path+"xgb/decision_boundary_xgb")
        #-----------------------------------------------------------------------





        # Calculate cumulative return using best parameters
        df_cum_train_xgb, df_cum_test_xgb, model_xgb = \
            predict_and_calculate_cum_return(
                model=XGBClassifier(**anova_results_xgb['best_params']),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)

        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_xgb, df_cum_test_xgb, label_reg,
            group_label={0:'T1', 1:'T2', 2:'T3'},
            figsize=(8, 6), filename=output_path+"xgb/return_xgb",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_xgb], return_label=[0, 1, 2],
            list_labels=["Logistic regression"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path+"xgb/return_xgb_diff",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(
            model=model_xgb, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            filename=output_path+"xgb/decision_boundary_xgb")


    #---------------------------------------------------------------------------
    # kNN
    #---------------------------------------------------------------------------
    if run_knn:
        print("Running kNN")
        # Set parameters to search
        param_grid = {'n_neighbors': [int(x) for x in np.logspace(1, 4, 10)]}

        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_knn, cv_results_knn = grid_search_purged_cv(
                df_train=df_train,
                model=KNeighborsClassifier(),
                param_grid=param_grid,
                metric=metric,
                features=features, label=label,
                k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_knn.to_csv(output_path+"cv_results_knn.csv")
        else:
            best_params_knn = {'n_neighbors': 1000}

        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_knn, filename=output_path+"cv_results_knn",
            n_bins=10, figsize=(8, 5), alpha=0.6, hist_type='step')

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
            figsize=(8, 6), filename=output_path+"return_knn",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_knn], return_label=[0, 1, 2],
            list_labels=["Logistic regression"], label_reg=label_reg,
            figsize=(8, 6), filename=output_path+"return_knn_diff",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(
            model=model_knn, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            filename=output_path+"decision_boundary_knn")

    #---------------------------------------------------------------------------
    # Neural network
    #---------------------------------------------------------------------------
    if run_nn:
        print("Running neural network")
        # Set parameters to search
        param_grid = {
            'activation':['relu'],
            'hidden_layer_sizes':[
                (300, 300, 300, 300, 300),
                (500, 400, 300, 200, 100),
                (1000, 500, 400, 300, 200, 100, 50, 10)],
            'alpha': np.logspace(-7, -2, 5),
            'early_stopping':[True],
            'max_iter':[200],
            'learning_rate':['adaptive'] #['constant', 'adaptive']
            }
        
        # Perform hyperparameter search using purged CV
        if run_grid_search:
            best_params_nn, cv_results_nn = grid_search_purged_cv(
                df_train=df_train,
                model=MLPClassifier(),
                param_grid=param_grid,
                metric=metric,
                features=features, label=label,
                k=k, purge_length=3, verbose=False)
            # Save cross-validation results
            cv_results_nn.to_csv(output_path+"cv_results_nn.csv")
        else:
            best_params_nn = {
                'activation': 'relu', 'hidden_layer_sizes': (100, 100, 100),
                'alpha': 0.046415888336127725, 'early_stopping': True,
                'learning_rate': 'adaptive'}

        # Plot distribution of cross-validation results
        plot_cv_dist(
            cv_results_nn, filename=output_path+"cv_results_nn", n_bins=10,
            figsize=(8, 5), alpha=0.6, hist_type='step')

        # Calculate cumulative return using best parameters
        df_cum_train_nn, df_cum_test_nn, model_nn = \
            predict_and_calculate_cum_return(
                model=MLPClassifier(**best_params_nn),
                df_train=df_train, df_test=df_test,
                features=features, label_cla=label, label_fm=label_fm)
        # Make cumulative return plot
        plot_cumulative_return(
            df_cum_train_nn, df_cum_test_nn, label_reg,
            group_label={0:'T1', 1:'T2', 2:'T3'},
            figsize=(8, 6), filename=output_path+"return_nn",
            kwargs_train={'ylim':(-1, 7)}, kwargs_test={'ylim':(-1, 3)})
        plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test_nn], return_label=[0, 1, 2],
            list_labels=["Logistic regression"], label_reg=label_reg,
            figsize=(8, 6),
            filename=output_path+"return_nn_diff",
            kwargs_train={'ylim':(-1, 5)}, kwargs_test={'ylim':(-1, 3)})

        # Plot decision boundary of trained model
        plot_decision_boundary(
            model=model_nn, df=df_test, features=features, h=0.01,
            x_label=feature_x, y_label=feature_y,
            vlines=tertile_boundary[feature_x],
            hlines=tertile_boundary[feature_y], colors=colors,
            xlim=(-3, 3), ylim=(-3, 3), figsize=(8, 6), ticks=[0, 1, 2],
            filename=output_path+"decision_boundary_nn")

        
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
        df_cum_test_tf = pickle.load(
            open(output_path+'df_cum_test_tf.pickle', 'rb'))


        plot_cumulative_return_diff(
            list_cum_returns=[
                df_cum_test_lr, df_cum_test_knn,
                df_cum_test_xgb, df_cum_test_tf],
            list_labels=["Linear", "KNN", "XGB", "NN"],
            label_reg=label_reg,
            figsize=(8, 6), return_label=[0, 1, 2],
            kwargs_train={'ylim':(-1, 3)},
            kwargs_test={'ylim':(-1, 3)},
            legend_order=["NN", "XGB", "Linear", "KNN"],
            filename=output_path+"return_diff_summary")

        # Save results
        save_summary(
            df_train, label, metric, output_path,
            cv_results={
                'Logistic': cv_results_lr,
                'XGB': cv_results_xgb,
                'KNN': cv_results_knn,
            })



    print("Successfully completed all tasks")


