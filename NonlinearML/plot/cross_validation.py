import pandas as pd

from NonlinearML.plot.plot import *
import NonlinearML.lib.utils as utils


#-------------------------------------------------------------------------------
# Cross-validation distribution
#-------------------------------------------------------------------------------
def plot_cv_dist(
    cv_results, filename, n_bins=10, figsize=(8,5), alpha=0.8, hist_type='step',
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

    # Loop over metrics to plot CV result distribution
    for name, values in zip(metric_names, metric_values):
        # Convert columns of list to multiple columns
        df_values = utils.expand_column(cv_results, values)

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
			hue_str={x:x for x in df_values['level_0'].unique()},
            hist_type=hist_type, x_range=x_range, legend_loc=legend_loc,
            legend_box=legend_box,
            norm=False, n_bins=n_bins, figsize=figsize,
            filename=filename+"_"+name, alpha=alpha, **kwargs)
    return


def plot_cv_box(cv_results, filename, figsize=(8,5), **kwargs):
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

    # Loop over metrics to plot CV result distribution
    for name, values in zip(metric_names, metric_values):
        # Convert columns of list to multiple columns
        df_values = utils.expand_column(cv_results, values)

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




def plot_cv_line(
	cv_results, filename, figsize=(8,5),
	legend_loc=None, legend_box=(0,-0.2), **kwargs):
    ''' Plot cross-validation results from each hyperparameter set.
    Performance is plotted vs position of kth fold.   
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

    # Loop over metrics to plot CV result distribution
    for name, values in zip(metric_names, metric_values):
        # Convert columns of list to multiple columns
        df_values = utils.expand_column(cv_results, values)
        df_values = df_values.transpose()

        # Make line plot
        plot_line_multiple_cols(
            filename=filename+"_"+name,
			df=df_values, x='index', list_y=list(df_values.columns),
            figsize=figsize, x_label='k-fold', y_label=name,
			legends=list(df_values.columns),
			legend_loc=None, legend_box=(0,-0.2), **kwargs)
    return

def plot_cv_correlation(
	cv_results, filename, figsize=(8,5), mask=False, **kwargs):
    ''' Plot correlation among different metrics such as r2, MSE, MAPE, etc.
    Args:
        cv_results: Cross-validation result from grid_search_purged_cv or
            grid_search_cv function
        others: Plotting options
    Returns:
        None
    '''
    # Extract list of metrics
    metric_train_names = [
        metric[:-7] for metric in cv_results.columns
            if 'train_values' in metric]
    metric_train_values = [
        metric for metric in cv_results.columns
            if 'train_values' in metric]
    metric_val_names = [
        metric[:-7] for metric in cv_results.columns
            if 'val_values' in metric]
    metric_val_values = [
        metric for metric in cv_results.columns
            if 'val_values' in metric]

    # Loop over metrics to correct values in training set
    df_corr_train = pd.DataFrame()
    for name, values in zip(metric_train_names, metric_train_values):
        df_corr_train[name] = utils.expand_column(cv_results, values)\
            .stack().reset_index()[0]

    # Loop over metrics to correct values in validation set
    df_corr_val = pd.DataFrame()
    for name, values in zip(metric_val_names, metric_val_values):
        df_corr_val[name] = utils.expand_column(cv_results, values)\
            .stack().reset_index()[0]
    # Create mask so that correlation plot only shows bottom half
    if mask==True:
        mask = np.zeros(df_corr_train.corr().shape, dtype=bool)
        mask[np.triu_indices(len(mask))] = True

    # Make heatmap
    plot_heatmap(
        df=df_corr_train.corr(),
        x_label='Metrics', y_label='Metrics', figsize=(8,6),
        annot_kws={'fontsize':10}, annot=True, fmt='.3f', vmin=-1, vmax=1,
        filename=filename+"_train", mask=mask, cmap='RdYlBu', **kwargs)

    plot_heatmap(
        df=df_corr_val.corr(),
        x_label='Metrics', y_label='Metrics', figsize=(8,6),
        annot_kws={'fontsize':10}, annot=True, fmt='.3f', vmin=-1, vmax=1,
        filename=filename+"_val", mask=mask, cmap='RdYlBu', **kwargs)

    return



