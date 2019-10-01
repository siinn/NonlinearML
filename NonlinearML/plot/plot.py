import numpy as np
import itertools

import NonlinearML.lib.io as io
from NonlinearML.lib.utils import create_folder
from NonlinearML.plot.style import load_matplotlib, lines
from NonlinearML.plot.style import load_matplotlib, load_seaborn, lines
plt = load_matplotlib()
sns = load_seaborn()



#-------------------------------------------------------------------------------
# plotting functions
#-------------------------------------------------------------------------------
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
    # Create output folder and save figure
    create_folder(filename)
    if filename != "":
        io.message('Saving figure as "%s.png"' %filename)
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
    # customize plot
    ax = ax.reshape(n_subplot_rows, n_subplot_columns)
    # Create output folder and save figure
    create_folder(filename)
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
    # Create output folder and save figure
    create_folder(filename)
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
    line = lines() 
    for name, df_group in df.groupby(by=groupby, sort=True):
        if x=="index":
            df_group[y].plot(
                kind='line', legend=True, label=group_label[name],
                linewidth=2.0, linestyle=next(line), **kwargs)
        else:
            df_group.set_index(x)[y].plot(
                kind='line', legend=True, label=group_label[name],
                linewidth=2.0, linestyle=next(line), **kwargs)
    # customize plot
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
    # Create output folder and save figure
    create_folder(filename)
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
    line = lines() 
    for i, y in enumerate(list_y):
        if x=="index":
            df[y].plot(
                kind='line', linewidth=2.0, label=legends[i],
                linestyle=next(line), **kwargs)
        else:
            df.set_index(x)[y].plot(
                kind='line', linewidth=2.0, label=legends[i],
                linestyle=next(line), **kwargs)
    # customize plot
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    if ylog:
        ax[0].set_yscale('log')
    # Create output folder and save figure
    create_folder(filename)
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
    # customize plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    # Create output folder and save figure
    create_folder(filename)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)


def plot_scatter(
    df, x, y, x_label="", y_label="", figsize=(20,6), filename="",
    legend=False, leg_loc='center left', legend_title=None, bbox_to_anchor=(1, 0.5),
    ylim=False, xlim=False, **kwargs):
    ''' create scatter plot from given dataframe
    Args:
        df: Pandas dataframe
        others: plotting options
    Returns:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # make scatter plot
    ax = sns.scatterplot(data=df, x=x, y=y, **kwargs)
    # customize plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if legend:
        ax.legend(loc=leg_loc, bbox_to_anchor=bbox_to_anchor, title=legend_title)
        #Hack to remove the first legend entry (which is the undesired title)
        vpacker = ax.get_legend()._legend_handle_box.get_children()[0]
        vpacker._children = vpacker.get_children()[1:]
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    # Create output folder and save figure
    create_folder(filename)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)
    plt.cla()
    return

def plot_reg(
    df, x, y, x_label="", y_label="", figsize=(20,6), filename="",
    legend=False, leg_loc='center left', legend_title=None, bbox_to_anchor=(1, 0.5),
    ylim=False, xlim=False, **kwargs):
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
    ax = sns.regplot(data=df, x=x, y=y, **kwargs)
    # customize plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if legend:
        ax.legend(loc=leg_loc, bbox_to_anchor=bbox_to_anchor, title=legend_title)
        #Hack to remove the first legend entry (which is the undesired title)
        vpacker = ax.get_legend()._legend_handle_box.get_children()[0]
        vpacker._children = vpacker.get_children()[1:]
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    # Create output folder and save figure
    create_folder(filename)
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
    # customize plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    plt.legend(loc='upper left')
    plt.tight_layout()
    fig.suptitle('')
    if filename != "":
        # Create output folder and save figure
        create_folder(filename)
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
              + df_err_list[group_name].applymap(lambda x: ' (%.3f)' % float(x))
        else:
            df_annot = True
        sns.heatmap(df_list[group_name], ax=ax[i],
                    annot=df_annot,\
                    cmap=cmap, **kwargs)
        # customize plot
        ax[i].set_ylabel(y_label)
        ax[i].set_xlabel(x_label)
        if group_map:
            ax[i].set_title(group_map[group_name])
    # customize plot
    plt.tight_layout()
    if filename != "":
        # Create output folder and save figure
        create_folder(filename)
        plt.savefig('%s.png' % filename)
    return

