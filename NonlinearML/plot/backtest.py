import pandas as pd

import NonlinearML.lib.io as io

import NonlinearML.lib.backtest as backtest
from NonlinearML.lib.utils import create_folder
from NonlinearML.plot.plot import *
from NonlinearML.plot.style import load_matplotlib
plt = load_matplotlib()




#-------------------------------------------------------------------------------
# Cumulative return plots
#-------------------------------------------------------------------------------
def plot_cumulative_return(
    df_cum_train, df_cum_test, label_reg, filename, figsize=(15,6),
    group_label={0:"Q1", 1:"Q2", 2:"Q3"}, date_column="eom",
    train_ylim=(-1,7), test_ylim=(-1,5),
    kwargs_train={}, kwargs_test={}):
    ''' Wrapper of plotting functions. Create cumulative return plot for train
    and test dataset.
    Args:
        df_cum_train: cumulative return obtained from train set
        df_cum_test: cumulative return obtained from test set
        label_reg: name of target label. Only used as axis label.
        group_label: dictionary to map between label and recognizable string
        time: time column
        filename: filename
        others: kwargs for plotting options
    Returns:
        None
    ''' 
    io.message(" > Plotting cumulative return plots with filename:")
    io.message("\t"+filename)
    # plot train dataset
    plot_line_groupby(
        df=df_cum_train.sort_values(date_column),
        x=date_column, y="cumulative_return",
        groupby="pred",
        group_label = {key:group_label[key]+" (Train)" for key in group_label}, 
        x_label="Time", y_label="Cumulative %s" %label_reg, ylog=False,
        ylim=train_ylim,
        figsize=figsize, filename = "%s_train" %filename, **kwargs_train)
    
    # plot test dataset
    plot_line_groupby(
        df=df_cum_test.sort_values(date_column),
        x=date_column, y="cumulative_return",
        groupby="pred",
        group_label = {key:group_label[key]+" (Test)" for key in group_label},\
        x_label="Time", y_label="Cumulative %s" %label_reg, ylog=False,
        ylim=test_ylim,
        figsize=figsize, filename = "%s_test" %filename, **kwargs_test)
    return


def plot_cumulative_return_diff(
    list_cum_returns, list_labels, label_reg, return_label=['Q1', 'Q2', 'Q3'],
    figsize=(15,6), filename="", date_column='eom', ylim=(-1,7),
    legend_order=None, **kwargs):
    """ Wrapper of plotting function. This function plots difference in
    cumulative return between top and bottom classes.
    Args:
        list_cum_Returns: list of dataframe representing cumulative returns
            (output of "calculate_cum_return")
        list_label: list of model names. Ex. ['xgb']
        label_reg: Regression label. Ex. 'fqTotalReturn'
        return_label: Classification label in descending order.
            Ex. ['high', 'medium', 'low']
    """
    # Calculate difference in return and concatenate
    df_diff = pd.concat([
        backtest.calculate_diff_return(
            cum_return, return_label=return_label,
            output_col=label, time=date_column)
        for cum_return, label in zip(list_cum_returns, list_labels)])

    # Sort by dates
    df_diff = df_diff.sort_values(date_column)

    # If legend order is given, pass it to plot_line_groupby. 
    if legend_order:
        # plot test dataset
        plot_line_groupby(
            df=df_diff, legend_order=legend_order,
            x="index", y="cumulative_return",
            groupby="pred",
            group_label = {key:key for key in df_diff["pred"].unique()},
            x_label="Time",
            y_label="Cumulative %s\nTop - bottom" %label_reg,
            ylog=False, figsize=figsize, ylim=ylim,
            filename = "%s" %filename, **kwargs)
    else:
        # plot test dataset
        plot_line_groupby(
            df=df_diff,
            x="index", y="cumulative_return", groupby="pred",
            group_label = {key:key for key in df_diff["pred"].unique()},
            x_label="Time",
            y_label="Cumulative %s\nTop - bottom" %label_reg,
            ylog=False, figsize=figsize, ylim=ylim,
            filename = "%s" %filename, **kwargs)
    return        


#-------------------------------------------------------------------------------
# Partial dependence plots
#-------------------------------------------------------------------------------
def plot_partial_dependence_1D(
    model, examples, target_names, feature_interest,
    grid_resolution=20, with_variance=True, figsize=(8,5),
    colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"],
    ylim=None, xlim=None, ylabel="Probability",
    filename="plots/pdp", merge_plots=True):
    ''' Create 1D partial dependence plots for each class using Skater library.
    Details can be found below.
        https://github.com/oracle/Skater/
    Args:
        model: pretrained sklearn model.
               Or any model that has .predict_prob method.
        examples: dataframe containing features.
        target_names: name of classes. ex. ['T1', 'T2', 'T3'].
        feature_interest: feature of interest.
        grid_resolution: resolution of partial dependence plot.
        others: plotting options.
    Returns:
        None
    '''
    io.message("Plotting partial dependence plot with filename:")
    io.message("\t"+filename)
    # Create figures for partial dependence plot using Skater
    interpreter = Interpretation(
        examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(
        model.predict_proba, examples=examples, target_names=target_names)
    axes_list = interpreter.partial_dependence.plot_partial_dependence(
        feature_ids=[feature_interest],
        modelinstance=im_model, 
        grid_resolution=grid_resolution, 
        with_variance=with_variance,
        figsize=figsize)
    # Get half length of output axes list
    half_length = int(len(axes_list[0]) / 2)

    if merge_plots:
        # Define dictionary to hold pdp values
        pdp_x = {}
        pdp_y = {}

        # Customize plots and save as png 
        for i, ax in enumerate(axes_list[0][half_length:]):
            # Extract legend and handle
            handle = ax.get_legend_handles_labels()
            legend = handle[1][0]
            # Get values from axes and append to dictionary
            pdp_x[legend] = handle[0][0].get_data()[0]
            pdp_y[legend] = handle[0][0].get_data()[1]

        # Combine pdp into a single plot
        fig, ax = plt.subplots(1,1, figsize=figsize)
        # Draw lines
        for i, legend in enumerate(target_names):
            ax.plot(pdp_x[legend], pdp_y[legend], label=legend, color=colors[i])
        # Customize plot
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(feature_interest)
        ax.ticklabel_format(style='plain')
        ax.legend()
        # Create output folder and save figure
        create_folder(filename)
        fig.tight_layout()
        fig.savefig("%s.png" % (filename))
                                                                    
    else:
        # Customize plots and save as png 
        for i, ax in enumerate(axes_list[0][half_length:]):
            if ylim:
                ax.set_ylim(ylim)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(style='plain')
            # Create output folder and save figure
            create_folder(filename)
            fig = ax.figure
            fig.tight_layout()
            fig.savefig("%s_%s.png" % (filename, i))



def plot_partial_dependence_2D(
    model, examples, target_names, feature_interest, feature_other,
    grid_resolution=20, with_variance=True, figsize=(8,5),
    zlim=None, zlabel="Probability", filename="plots/pdp_2D"):
    ''' Create 2D partial dependence plots for each class using Skater library.
        Details can be found below.
        https://github.com/oracle/Skater/
    Args:
        model: pretrained sklearn model.
               Or any model that has .predict_prob method.
        examples: dataframe containing features.
        target_names: name of classes. ex. ['T1', 'T2', 'T3'].
        feature_interest: feature of interest.
        feature_other: another features to compare with the feature of interest.
        grid_resolution: resolution of partial dependence plot.
        others: plotting options.
    Returns:
        None
    '''
    io.message("Plotting 2D partial dependence plot with filename: \n >%s" %filename)
    # Create figures for partial dependence plot using Skater
    interpreter = Interpretation(
        examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(
        model.predict_proba, examples=examples, target_names=target_names)
    axes_list = interpreter.partial_dependence.plot_partial_dependence(
        feature_ids=[
            # Pass feature_ids as list of tuples. ex. [('AG', 'FCFA')]
            tuple([feature_interest, feature_other])], 
        modelinstance=im_model, 
        grid_resolution=grid_resolution, 
        with_variance=with_variance,
        figsize=figsize)

    # Get half length of output axes list
    half_length = int(len(axes_list[0]) / 2)

    # Customize plots and save as png 
    for ax in axes_list[0][half_length:]:
        # Retrieve subplots and label
        ax_3d = ax.get_figure().axes[0]
        ax_2d = ax.get_figure().axes[1]
        label=ax.get_zlabel().strip()
        # This axes has two sub axes. Change their ticklabel to
        # avoid scientific notation
        for sub_ax in ax.get_figure().axes:
            sub_ax.ticklabel_format(style='plain')
        # Customize 3D plot
        if zlim:
            ax_3d.set_ylim(zlim)
        ax_3d.set_zlabel(zlabel)
        # Customize 2D plot
        ax_2d.set_title("")
        # Save figure
        # Create output folder and save figure
        create_folder(filename)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("%s_%s.png" % (filename, label))


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

