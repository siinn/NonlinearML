import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
#from skater.core.explanations import Interpretation
#from skater.model import InMemoryModel
import itertools


# set plot style
markers=('x', 'p', "|", '*', '^', 'v', '<', '>')
lines=("-","--","-.",":")
#hatchs = ('/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*')
hatchs = ('/', '\\', '|', '-', '+', 'x')

# Set cyler
''' Example: linestyle=next(line_cycler), hatch=next(hatch_cycler) '''
marker_cyler = itertools.cycle(markers)
line_cycler = itertools.cycle(lines)
hatch_cycler = itertools.cycle(hatchs)


mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('fivethirtyeight')
#plt.style.use('tableau-colorblind10')
#plt.style.use('seaborn-dark')
#plt.style.use('ggplot')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'




#-------------------------------------------------------------------------------
# plotting functions
#-------------------------------------------------------------------------------
def plot_learning_curve(
    df, xcol="h", ycols=["f1_train", "f1_test"], title="", figsize=(8,8)):
    ''' plot learning curve and save as png
    Args:
        df: dataframe containing model score and training length
        xcol: column representing training length
        ycols: list of column names representing model scores
        others: plotting options
    Returns: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # make plot
    line = itertools.cycle(lines) 
    for i, ycol in enumerate(ycols):
        df.plot.line(x=xcol, y=ycol, ax=ax, legend=True, linestyle=next(line))
    # customize plot and save
    ax.set_ylabel("Average F1 score")
    ax.set_xlabel("Training length (months)")
    ax.set_ylim(0,0.4)
    plt.tight_layout()
    plt.savefig('plots/learning_curve_%s.png' % title)
    return

def plot_distribution(
    df, columns, n_rows, n_columns, bins=[], ylog=[], xrange=[], ylim=[],
    title=[], x_label=[], y_label=[], figsize=(10,10), filename="",
    show_sigma=False, vlines=None, **kwargs):
    ''' plot distributions of given columns in a grid of n_rows x n_columns
    subplots.
    Args:
        df: Pandas dataframe
        columns: list of columns of of interest
        show_sigma: show two vertical lines at mean +- 2*sigma
        vlines: Dataframe used to draw vertical line at observation.
            Used for checking if observations are within a confidence interval.
        others: plotting options
    Returns:
        None
    '''
    def _get_default(args, default, i):
        if not args:
            return default
        else:
            return args[i]
    # create figure and axis
    fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    for i, col in enumerate(columns):
        # set parameters if not specified
        _bin = _get_default(bins, 100, i)
        _xrange = _get_default(xrange, (df[col].min(), df[col].max()), i)
        _title = _get_default(title, col, i)
        _xlabel = _get_default(x_label, col, i)
        _ylabel = _get_default(y_label, "n", i)
        _log = _get_default(ylog, False, i)
        # make plot
        df[col].plot(
            bins=_bin, ax=ax[i], kind='hist', histtype='step', linewidth=2,
            range=_xrange, **kwargs)
        ax[i].set_xlabel(_xlabel)
        ax[i].set_ylabel(_ylabel)
        ax[i].set_title(_title)
        if _log:
            ax[i].set_yscale('log')
        if ylim:
            ax[i].set_ylim(ylim[i])
        if show_sigma:
            # draw vertical lines showing +- 2 sigma
            ax[i].axvline(
                x=df[col].mean() + 2*df[col].std(), color='red',
                label='+2sigma', linestyle='--')
            ax[i].axvline(
                x=df[col].mean() - 2*df[col].std(), color='red',
                label='+2sigma', linestyle='--')
            if type(vlines) != None:
                ax[i].axvline(
                    x=vlines[col], color='blue', label='observation',
                    linestyle='-')
    # remove extra subplots
    for x in np.arange(len(columns),len(ax),1):
        fig.delaxes(ax[x])
    plt.tight_layout()
    if filename != "":
        print('Saving figure as "%s.png"' %filename)
        plt.savefig('%s.png' % filename)
    return

def plot_dist_groupby_hue(
    df, x, group_var, group_title, hue, hue_str, norm=False,
    n_subplot_columns=1, n_bins=50, figsize=(20,16), filename="", **kwargs):
    ''' plot distribution of given variable for each group. Seperate plot will
    be generated for each group. 

    Args:
        df: Pandas dataframe
        x: variable to plot
        group_var: categorical variable for group
        group_title: dictionary that maps group variable to human-recognizable
            title (45 -> Information technology)
        hue: additional category. seperate distribution will be plotted for
            each hue within the same group plot.
        hue_str: dictionary to map hue value and name.
            Examples:
            0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Returns: None
    '''
    n_groups = df[group_var].nunique()
    # create figure and axes
    n_subplot_rows = round(n_groups / n_subplot_columns)
    fig, ax = plt.subplots(
        n_subplot_rows, n_subplot_columns, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    for i, group_name in enumerate(sorted(df[group_var].unique())):
        # filter group
        df_group = df.loc[df[group_var] == group_name]
        n_hue = df[hue].nunique()
        # loop over hue
        for j, hue_name in enumerate(sorted(df_group[hue].unique())):
            df_hue = df_group.loc[df_group[hue] == hue_name]
            df_hue[x].hist(
                bins=n_bins, alpha=0.6, ax=ax[i], edgecolor="black",
                label=hue_str[hue_name], density=norm, **kwargs)
        # customize plot
        ax[i].set_xlabel(x)
        ax[i].set_ylabel("n")
        ax[i].set_title(group_title[i])
        ax[i].grid(False)
        ax[i].legend()
    # customize and save plot
    ax = ax.reshape(n_subplot_rows, n_subplot_columns)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)
    plt.cla()
    return

def plot_dist_hue(
    df, x, hue, hue_str, hist_type='step', ylabel='n', norm=False, n_bins=50,
    x_range=None, legend_loc=None, legend_box=(0,0), figsize=(8,5),
    filename="", **kwargs):
    ''' plot distribution of given variable for each group.
    Args:
        df: Pandas dataframe
        x: variable to plot
        hue: additional category. seperate distribution will be plotted for
            each hue within the same plot.
        hue_str: dictionary to map hue value and name.
            Example:
                0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Returns: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # get xrange
    if x_range==None:
        x_range = (df[x].min(), df[x].max())
    # filter group
    n_hue = df[hue].nunique()
    # loop over hue
    for j, hue_name in enumerate(sorted(df[hue].unique())):
        df_hue = df.loc[df[hue] == hue_name]
        df_hue[x].hist(
            bins=n_bins, histtype=hist_type, ax=ax, label=hue_str[hue_name],
            density=norm, range=x_range, linewidth=2.0, **kwargs)
    # customize plot
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if legend_loc==None:
        ax.legend()
    else:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_box)
    # customize and save plot
    plt.tight_layout()
    plt.savefig('%s.png' % filename)
    plt.cla()


def plot_line_groupby(
    df, x, y, groupby, group_label, ylog=False, x_label="", y_label="",
    figsize=(20,6), filename="", xlim=None, ylim=None, legend_order=None,
    **kwargs):
    ''' create line plot for different group in the same axes.
    Args:
        df: Pandas dataframe
        x: column used for x
        y: column to plot
        groupby: column representing different groups
        group_label: dictionary that maps gruop value to title.
            Example: {0:"AG1", 1:"AG2", etc.}
        others: plotting options
    Returns:
        None
    '''
    def _get_handle(name, labels, handles):
        ''' Get handle that matches with given name.'''
        for label, handle in zip(labels, handles):
            if label == name:
                return handle
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax=ax.flatten()
    line = itertools.cycle(lines) 
    for name, df_group in df.groupby(by=groupby, sort=False):
        if x=="index":
            df_group[y].plot(
                kind='line', legend=True, label=group_label[name],
                linewidth=2.0, linestyle=next(line), **kwargs)
        else:
            df_group.set_index(x)[y].plot(
                kind='line', legend=True, label=group_label[name],
                linewidth=2.0, linestyle=next(line), **kwargs)
    # customize and save plot
    if ylog:
        ax[0].set_yscale('log')
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    if xlim:
        ax[0].set_xlim(xlim)
    if ylim:
        ax[0].set_ylim(ylim)
    # order legends
    if legend_order:
        handles, labels = ax[0].get_legend_handles_labels()
        plt.legend(
            [_get_handle(name, labels, handles) for name in legend_order],
            legend_order)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)
    plt.cla()


def plot_line_multiple_cols(
    df, x, list_y, legends, x_label, y_label, ylog=False, figsize=(20,6),
    filename="", **kwargs):
    ''' Create line plot from multiple columns in the same axes.
    Args:
        df: Pandas dataframe
        x: column used for x
        list_y: list of column names to plot
        others: plotting options
    Returns:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax=ax.flatten()
    line = itertools.cycle(lines) 
    for i, y in enumerate(list_y):
        if x=="index":
            df[y].plot(
                kind='line', linewidth=2.0, label=legends[i],
                linestyle=next(line), **kwargs)
        else:
            df.set_index(x)[y].plot(
                kind='line', linewidth=2.0, label=legends[i],
                linestyle=next(line), **kwargs)
    # customize and save plot
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    if ylog:
        ax[0].set_yscale('log')
    #ax[0].grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s.png' % filename)
    plt.cla()
    return
    
def plot_heatmap(
    df, x_label, y_label, figsize=(20,6), filename="", cmap="Blues", **kwargs):
    ''' create heatmap from given dataframe
    Args:
        df: Pandas dataframe
        others: plotting options
    Returns:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # plot heatmap
    ax = sns.heatmap(df, cmap=cmap, **kwargs)
    # customize and save plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)


def plot_scatter(
    df, x, y, x_label="", y_label="", figsize=(20,6), filename="", **kwargs):
    ''' create scatter plot from given dataframe
    Args:
        df: Pandas dataframe
        others: plotting options
    Returns:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # plot heatmap
    ax = sns.scatterplot(data=df, x=x, y=y, **kwargs)
    # customize and save plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('%s.png' % filename)
    plt.cla()
    return

def plot_box(
    df, x, y, title, x_label, y_label, ylim=None, figsize=(20,6),
    filename="", **kwargs):
    ''' create box plot from pandas dataframe.
    Args:
        df: Pandas dataframe
        x: column used for x axis
        y: column used for y axis
        others: plotting options
    Returns:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # make plot
    ax = sns.boxplot(x=x, y=y, data=df,
                     boxprops={"edgecolor":"black"},
                     flierprops={"markeredgecolor":"black", "marker":"."},
                     whiskerprops={"color":"black"},
                     meanprops={"color":"blue"},
                     medianprops={"color":"black"},
                     capprops ={"color":"black"},
                     **kwargs)
    # customize and save plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    plt.legend(loc='upper left')
    plt.tight_layout()
    fig.suptitle('')
    if filename != "":
        plt.savefig('%s.png' % filename)
    return


def plot_heatmap_group(
    df_list, n_subplot_columns, x_label, y_label, df_err_list=None,
    group_map=None, ylim=None, figsize=(20,6), filename="", cmap="Blues",
    **kwargs):
    ''' create heatmap from given dataframe
    Args:
        df: list of dataframe
        x: column used for x axis
        y: column used for y axis
        n_subplot_columns: number of columns of subplot grid
        df_err_list: dataframe presenting uncertainty (optional)
        group: column used to separate plots into multiple subplots
        group_map: dictionary that maps values in group column to string.
            Example: {10:"Sector1", etc.}
        others: plotting options
    Returns:
        None
    '''
    # create figure and axes
    n_groups = len(df_list)
    n_subplot_rows = round(n_groups / n_subplot_columns)
    fig, ax = plt.subplots(n_subplot_rows, n_subplot_columns, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    # plot heatmap
    for i, group_name in enumerate(df_list.keys()):
        if df_err_list:
            df_annot =\
                df_list[group_name].applymap(lambda x: '%.3f' % float(x))\
                + df_err_list[group_name].applymap(
                    lambda x: ' (%.3f)' % float(x))
        else:
            df_annot = True
        sns.heatmap(df_list[group_name], ax=ax[i],
                    annot=df_annot,\
                    cmap=cmap, **kwargs)
        # customize and save plot
        ax[i].set_ylabel(y_label)
        ax[i].set_xlabel(x_label)
        if group_map:
            ax[i].set_title(group_map[group_name])
    # customize and save plot
    plt.tight_layout()
    if filename != "":
        plt.savefig('%s.png' % filename)
    return



#-------------------------------------------------------------------------------
# plotting functions specific to AG project
#-------------------------------------------------------------------------------
from Asset_growth.lib.backtest import *

def plot_decision_boundary(
    model, df, features, h=0.01, x_label="", y_label="", xlim=False, ylim=False,
    title=False, title_loc='center', annot=False, vlines = [], hlines = [],
    colors=["#BA2832", "#F3F2F2", "#2A71B2"], figsize=(8,4), ticks=[],
    filename=""):
    ''' Plot decision boundary of trained model.
    Args:
        model: Fitted model that has .predict method
        df: input dataframe containing two features. Used to extract feature
            domain (mix, max values). Column order decides x and y. First
            feature becomes x, second feature becomes y. 
        features: list of features. ex. ["AG", "FCFA"]
        h: step size when creating grid
        vlines, hlines: vertical and horizontal lines to draw given in list
            Example: vlines=[-0.5, 0, 0.5]
        others: plotting option
    Returns:
        None
    '''
    print("Plotting decision boundary with filename: %s" %filename)
    # Get x and y domain
    if xlim:
        x_min, x_max = xlim
    else:
        x_min, x_max = (df[features].iloc[:,0].min(),
                        df[features].iloc[:,0].max())
    if ylim:
        y_min, y_max = ylim
    else:
        y_min, y_max = (df[features].iloc[:,1].min(),
                        df[features].iloc[:,1].max())
    # Create grid of (x,y) pairs.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    df_mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])\
                .rename({i:features[i] for i in range(len(features))}, axis=1)
    # Make prediction for each point on grid
    z = model.predict(df_mesh)
    z = z.reshape(xx.shape)
    # Put the result into a color plot
    cmap = ListedColormap(sns.color_palette(colors).as_hex())
    fig, ax = plt.subplots(1,1, figsize=figsize)
    im = ax.pcolormesh(xx, yy, z, cmap=cmap)
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, loc=title_loc)
    if annot:
        ax.text(
            annot['x'], annot['y'], annot['text'],
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(xx.min(), xx.max())
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(yy.min(), yy.max())
    plt.tight_layout()
    colorbar = fig.colorbar(im, ax=ax)
    if ticks:
        colorbar.set_ticks(ticks)
    # Draw vertical and horizontal lines.
    if vlines:
        for x in vlines:
            plt.axvline(x, linewidth=1, linestyle='--', color='black')
    if hlines:
        for y in hlines:
            plt.axhline(y, linewidth=1, linestyle='--', color='black')
    # Save figure
    plt.savefig('%s.png' %filename)


def plot_decision_boundary_multiple_hparmas(param_grid, label, **kwargs):
    ''' Plot decision boundary for each hyperparameter set. This is a wrapper
        of 'plot_decision_boundary' function.
    Args:
        param_grid: Hyperparamater grid to search.
        label: classification label
        **kwargs:
            Arguments for 'plot_decision_boundary'. Only difference is
            that df must be training data.
            In 'plot_decision_boundary', this could be training or test data
            as it is only used to extract range of the plot.
    Returns:
        None
    '''
    print('Creating decision boundary plots for each combination of',
          ' hyperparameters')
    # Extract model from kwargs
    model = kwargs['model']
    df_train = kwargs['df']
    features = kwargs['features']
    filename = kwargs['filename']
    # Get all possible combination of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Loop over different values of k
    n_experiments = len(experiments)
    count=0
    print("debug 0")
    for i, params in enumerate(experiments):
        count = count + 1
        print(' > Experiment (%s/%s)' % (count, n_experiments))
        print(' > Parameters: %s' % str(params))
        # Train model with given hyperparameter set
        model=model.set_params(**params)
        model.fit(df_train[features], df_train[label])
        # Assign trained model back to kwargs
        kwargs['model'] = model
        # Add counter to filename
        kwargs['filename'] = filename + "_%s" % i
        # Plot decision boundary of trained model
        plot_decision_boundary(
            annot={
                'text':str(params).strip('{}').replace('\'','')\
                                                 .replace(',','\n')\
                                                 .replace('\n ', '\n'),
                'x':0.02, 'y':0.98},
            **kwargs)
    return



def plot_decision_boundary_pdp(
    model, examples, target_names, feature_interest, feature_other,
    grid_resolution=20, with_variance=True, figsize=(8,5),
    colors=["#3DC66D", "#F3F2F2", "#DF4A3A"],
    filename="plots/decesion_boundary_pdp", x_label="", y_label="",
    xlim=False, ylim=False, vlines = [], hlines = [], ticks=[]):
    ''' Create decision boundaries of two factors using partial dependence plot
    generated by Skater library. Details can be found below.
        https://github.com/oracle/Skater/
    Args:
        model: pretrained sklearn model. Or any model that has
            .predict_prob method.
        examples: dataframe containing features.
        target_names: name of classes. ex. ['T1', 'T2', 'T3'].
        feature_interest: feature of interest.
        feature_other: another features to compare with the feature of interest.
        grid_resolution: resolution of partial dependence plot.
        others: plotting options.
    Returns:
        None
    '''
    def _get_z(df, x, y, feature_interest, feature_other):
        '''Find row and return prediction given x and y.'''
        # Change data type to float
        return np.array(
            df.loc[
                (df[feature_interest]==x) & \
                (df[feature_other]==y)]['pred'])[0]
        
    print("Plotting decision boundary plot using partial dependence plot."\
           + " Filename: %s" %filename)
    # Create figures for partial dependence plot using Skater
    interpreter = Interpretation(
        examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(
        model.predict_proba, examples=examples, target_names=target_names)
    df_pd = interpreter.partial_dependence.partial_dependence(
        # Pass feature_ids as list of tuples. ex. [('AG', 'FCFA')]
        feature_ids=[feature_interest, feature_other], 
        modelinstance=im_model, 
        grid_resolution=grid_resolution, sample=False,
        grid_range=(0,1), progressbar=False)

    # Retrieve grid of (x,y) pairs.
    xx, yy = np.meshgrid(
        df_pd[feature_interest].unique(), df_pd[feature_other].unique())
    # Make prediction by selecting class with the highest probability
    df_pd['pred'] = df_pd[target_names].idxmax(axis=1)
    # Get z coordinate of decision boundary by searching for (x,y) from the dataframe.
    z = np.array(
        [
            [_get_z(df_pd,x,y, feature_interest, feature_other) for x in xx[0]]
            for y in yy[:,0]])
    # Put the result into a color plot
    cmap = ListedColormap(sns.color_palette(colors).as_hex())
    fig, ax = plt.subplots(1,1, figsize=figsize)
    im = ax.pcolormesh(xx, yy, z, cmap=cmap)
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(xx.min(), xx.max())
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(yy.min(), yy.max())
    plt.tight_layout()
    colorbar = fig.colorbar(im, ax=ax)
    if ticks:
        colorbar.set_ticks(ticks)
    # Draw vertical and horizontal lines.
    if vlines:
        for x in vlines:
            plt.axvline(x, linewidth=1, linestyle='--', color='black')
    if hlines:
        for y in hlines:
            plt.axhline(y, linewidth=1, linestyle='--', color='black')
    # Save figure
    plt.savefig('%s.png' %filename)



def plot_cumulative_return(
    df_cum_train, df_cum_test, label_reg, filename, figsize=(15,6),
    group_label={0:"Q1", 1:"Q2", 2:"Q3"}, time="eom",
    kwargs_train={}, kwargs_test={}):
    ''' Wrapper of plotting functions. Create cumulative return plot for train
    and test dataset.
    Args:
        df_cum_train: cumulative return obtained from train set
        df_cum_test: cumulative return obtained from test set
        label_reg: name of target label
        group_label: dictionary to map between label and recognizable string
        time: time column
        filename: filename
        others: kwargs for plotting options
    Returns:
        None
    ''' 
    print("Plotting cumulative return plots with filename: %s" %filename)
    # plot train dataset
    plot_line_groupby(
        df=df_cum_train, x=time, y="cumulative_return",
        groupby="pred",
        group_label = {key:group_label[key]+" (Train)" for key in group_label}, 
        x_label="Time", y_label="Cumulative %s" %label_reg, ylog=False,
        figsize=figsize, filename = "%s_train" %filename, **kwargs_train)
    
    # plot test dataset
    plot_line_groupby(
        df=df_cum_test, x=time, y="cumulative_return",
        groupby="pred",
        group_label = {key:group_label[key]+" (Test)" for key in group_label},\
        x_label="Time", y_label="Cumulative %s" %label_reg, ylog=False,
        figsize=figsize, filename = "%s_test" %filename, **kwargs_test)
    return


def plot_cumulative_return_diff(
    list_cum_returns, list_labels, label_reg, return_label=['Q1', 'Q2', 'Q3'],
    figsize=(15,6), filename="",
    kwargs_train={}, kwargs_test={}, legend_order=None):
    ''' Wrapper for plotting function. This function plots difference in
    cumulative return for given models where
        difference in return is defined as Q1+Q2 - Q3.
    Args:
        list_cum_Returns: list of dataframe representing cumulative returns
        (output of "predict_and_calculate_cum_return")
        list_label: list of labels for the models
        label_reg: regression label. ex. 'fqTotalReturn'
    '''
    # Calculate difference in return and concatenate
    df_diff_q1q2_q3 = pd.concat([calculate_diff_return(cum_return, return_label=return_label, output_col=label)[0]
                                 for cum_return, label in zip(list_cum_returns, list_labels)])
    df_diff_q1_q3 = pd.concat([calculate_diff_return(cum_return, return_label=return_label, output_col=label)[1]
                               for cum_return, label in zip(list_cum_returns, list_labels)])

    # If legend order is given, pass it to plot_line_groupby. 
    if legend_order:
        # plot test dataset
        plot_line_groupby(df=df_diff_q1q2_q3, legend_order=legend_order,\
                          x="index", y="cumulative_return", groupby="pred", group_label = {key:key for key in df_diff_q1q2_q3["pred"].unique()},\
                          x_label="Time", y_label="Cumulative %s\n(Q1+Q2) - Q3" %label_reg, ylog=False, figsize=figsize, filename = "%s_q1q2_q3" %filename, **kwargs_train)
        plot_line_groupby(df=df_diff_q1_q3, legend_order=legend_order,\
                          x="index", y="cumulative_return",\
                          groupby="pred", group_label = {key:key for key in df_diff_q1q2_q3["pred"].unique()},\
                          x_label="Time", y_label="Cumulative %s\nQ1 - Q3" %label_reg, ylog=False, figsize=figsize, filename = "%s_q1_q3" %filename, **kwargs_test)
    else:
        # plot test dataset
        plot_line_groupby(df=df_diff_q1q2_q3,\
                          x="index", y="cumulative_return", groupby="pred", group_label = {key:key for key in df_diff_q1q2_q3["pred"].unique()},\
                          x_label="Time", y_label="Cumulative %s\n(Q1+Q2) - Q3" %label_reg, ylog=False, figsize=figsize, filename = "%s_q1q2_q3" %filename, **kwargs_train)
        plot_line_groupby(df=df_diff_q1_q3,\
                          x="index", y="cumulative_return", groupby="pred", group_label = {key:key for key in df_diff_q1q2_q3["pred"].unique()},\
                          x_label="Time", y_label="Cumulative %s\nQ1 - Q3" %label_reg, ylog=False, figsize=figsize, filename = "%s_q1_q3" %filename, **kwargs_test)
    return        






def plot_partial_dependence_1D(model, examples, target_names, feature_interest,
                            grid_resolution=20, with_variance=True, figsize=(8,5), colors = ["#3DC66D", "#F3F2F2", "#DF4A3A"],
                            ylim=None, xlim=None, ylabel="Probability", filename="plots/pdp", merge_plots=True):
    ''' Create 1D partial dependence plots for each class using Skater library. Details can be found below.
        https://github.com/oracle/Skater/
    Args:
        model: pretrained sklearn model. Or any model that has .predict_prob method.
        examples: dataframe containing features.
        target_names: name of classes. ex. ['T1', 'T2', 'T3'].
        feature_interest: feature of interest.
        grid_resolution: resolution of partial dependence plot.
        others: plotting options.
    Returns:
        None
    '''
    print("Plotting partial dependence plot with filename: %s" %filename)
    # Create figures for partial dependence plot using Skater
    interpreter = Interpretation(examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(model.predict_proba, examples=examples, target_names=target_names)
    axes_list = interpreter.partial_dependence.plot_partial_dependence(feature_ids=[feature_interest],
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
        fig.tight_layout()
        fig.savefig("%s.png" % (filename))
                                                                    
    else:
        # Customize plots and save as png 
        for i, ax in enumerate(axes_list[0][half_length:]):
            if ylim:
                ax.set_ylim(ylim)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(style='plain')
            fig = ax.figure
            fig.tight_layout()
            fig.savefig("%s_%s.png" % (filename, i))



def plot_partial_dependence_2D(model, examples, target_names, feature_interest, feature_other,
                            grid_resolution=20, with_variance=True, figsize=(8,5),
                            zlim=None, zlabel="Probability", filename="plots/pdp_2D"):
    ''' Create 2D partial dependence plots for each class using Skater library. Details can be found below.
        https://github.com/oracle/Skater/
    Args:
        model: pretrained sklearn model. Or any model that has .predict_prob method.
        examples: dataframe containing features.
        target_names: name of classes. ex. ['T1', 'T2', 'T3'].
        feature_interest: feature of interest.
        feature_other: another features to compare with the feature of interest.
        grid_resolution: resolution of partial dependence plot.
        others: plotting options.
    Returns:
        None
    '''
    print("Plotting 2D partial dependence plot with filename: %s" %filename)
    # Create figures for partial dependence plot using Skater
    interpreter = Interpretation(examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(model.predict_proba, examples=examples, target_names=target_names)
    axes_list = interpreter.partial_dependence.plot_partial_dependence(feature_ids=[tuple([feature_interest, feature_other])], # Pass feature_ids as list of tuples. ex. [('AG', 'FCFA')]
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
        # This axes has two sub axes. Change their ticklabel to avoid scientific notation
        for sub_ax in ax.get_figure().axes:
            sub_ax.ticklabel_format(style='plain')
        # Customize 3D plot
        if zlim:
            ax_3d.set_ylim(zlim)
        ax_3d.set_zlabel(zlabel)
        # Customize 2D plot
        ax_2d.set_title("")
        # Save figure
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("%s_%s.png" % (filename, label))



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

