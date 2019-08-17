import pandas as pd
from NonlinearML.plot.plot import *


#-------------------------------------------------------------------------------
# Cross-validation distribution
#-------------------------------------------------------------------------------
def plot_cv_dist(
    cv_results, filename, n_bins=10, figsize=(8,5), alpha=0.6, hist_type='step',
    x_range=None, legend_loc=None, legend_box=(0,-0.2), **kwargs):
    ''' Plot cross-validation distribution from each hyperparameter set.
    For example, if CV is performed on 5 hyperparameter sets with k=10,
    this function will plot the distribution of 5 runs where each distribution
    has 10 entries.
    Args:
        cv_results: Cross-validation result from grid_search_purged_cv or
            grid_search_cv function
        others: Plotting options
    Returns:
        None
    '''
    # Extract list of metrics
    metric_names = [
        metric[:-7] for metric in cv_results.columns if 'values' in metric]
    metric_values = [
        metric for metric in cv_results.columns if 'values' in metric]

    # Convert string to list
    if type(cv_results[metric_values].iloc[0,0]) == str:
        cv_results[metric_values] = \
            cv_results[metric_values].applymap(
                lambda x: x.strip('][').split(', '))

    # Loop over metrics to plot CV result distribution
    for name, values in zip(metric_names, metric_values):
        # Convert columns of list to multiple columns
        df_values = pd.DataFrame(cv_results[values].tolist())
        df_values = df_values.astype(float)

        # Stack dataframe for plotting.
        ''' column 'level_0' represent different CV run (hyperparameter set).
            column name (ex. accuracy) represent values obtained from
            the CV run.
        '''
        df_values = df_values.stack().reset_index()
        df_values.rename({0:name}, axis=1, inplace=True)

        # Make histograms
        plot_dist_hue(
            df=df_values,
            x=name, ylabel='Single fold result',
            hue='level_0',
            hue_str={
                x:cv_results['params'].iloc[x]
                for x in df_values['level_0'].unique()},
            hist_type=hist_type, x_range=x_range, legend_loc=legend_loc,
            legend_box=legend_box,
            norm=False, n_bins=n_bins, figsize=figsize,
            filename=filename+"_"+name, alpha=alpha, **kwargs)
    return


def plot_cv_box(cv_results, filename, figsize=(8,5), cv_metric=None, **kwargs):
    ''' Plot cross-validation distribution from each hyperparameter set.
    For example, if CV is performed on 5 hyperparameter sets with k=10,
    this function will plot the distribution of 5 runs where each distribution
    has 10 entries.
    Args:
        cv_results: Cross-validation result from grid_search_purged_cv or
            grid_search_cv function
        others: Plotting options
    Returns:
        None
    '''
    # Extract list of metrics
    metric_names = [
        metric[:-7] for metric in cv_results.columns if 'values' in metric]
    metric_values = [
        metric for metric in cv_results.columns if 'values' in metric]

    # Convert string to list
    if type(cv_results[metric_values].iloc[0,0]) == str:
        cv_results[metric_values] = \
            cv_results[metric_values].applymap(
                lambda x: x.strip('][').split(', '))

    # Loop over metrics to plot CV result distribution
    for name, values in zip(metric_names, metric_values):
        if (cv_metric != None) and (cv_metric!=name):
            continue
        # Convert columns of list to multiple columns
        df_values = pd.DataFrame(cv_results[values].tolist())
        df_values = df_values.astype(float)

        # Stack dataframe for plotting.
        ''' column 'level_1' represent different CV run (hyperparameter set).
            column 'name' (ex. accuracy) represent values obtained from
            the CV run.
        '''
        df_values = df_values.transpose().stack().reset_index()
        df_values.rename({'level_1':"model", 0:name}, axis=1, inplace=True)

        # Make box plot
        plot_box(
            df=df_values,
            x='model', y=name, title="Model performance by %s" %name,
            x_label="Model", y_label=name,
            ylim=None, figsize=figsize,
            filename=filename+"_"+name, **kwargs)
    return

