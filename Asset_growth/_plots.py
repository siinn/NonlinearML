#----------------------------------------------
# plotting functions
#----------------------------------------------
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import seaborn as sns
import itertools

# set plot style
markers=('x', 'p', "|", '*', '^', 'v', '<', '>')
lines=("-","--","-.",":")
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

#----------------------------------------------

def plot_learning_curve(df, xcol="h", ycols=["f1_train", "f1_test"], title="", figsize=(8,8)):
    ''' plot learning curve and save as png
    Args:
        df: dataframe containing model score and training length
        xcol: column representing training length
        ycols: list of column names representing model scores
        others: plotting options
    Return: None
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

def plot_dist_groupby_hue(df, x, group_var, group_title, hue, hue_str, norm=False, n_subplot_columns=1, n_bins=50, figsize=(20,16), filename="", **kwargs):
    ''' plot distribution of given variable for each group. Seperate plot will be generated for each group. 
    Args:
        df: Pandas dataframe
        x: variable to plot
        group_var: categorical variable for group
        group_title: dictionary that maps group variable to human-recognizable title (45 -> Information technology)
        hue: additional category. seperate distribution will be plotted for each hue within the same group plot.
        hue_str: dictionary to map hue value and name. i.e. 0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Return: None
    '''
    n_groups = df[group_var].nunique()
    # create figure and axes
    n_subplot_rows = round(n_groups / n_subplot_columns)
    fig, ax = plt.subplots(n_subplot_rows, n_subplot_columns, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    for i, group_name in enumerate(sorted(df[group_var].unique())):
        # filter group
        df_group = df.loc[df[group_var] == group_name]
        n_hue = df[hue].nunique()
        # loop over hue
        for j, hue_name in enumerate(sorted(df_group[hue].unique())):
            df_hue = df_group.loc[df_group[hue] == hue_name]
            df_hue[x].hist(bins=n_bins, alpha=0.6, ax=ax[i], edgecolor="black", label=hue_str[hue_name], density=norm, **kwargs)
        # customize plot
        ax[i].set_xlabel(x)
        ax[i].set_ylabel("n")
        ax[i].set_title(group_title[i])
        ax[i].grid(False)
        ax[i].legend()
    # customize and save plot
    ax = ax.reshape(n_subplot_rows, n_subplot_columns)
    plt.tight_layout()
    plt.savefig('plots/dist_%s.png' % filename)
    plt.cla()
    return

def plot_dist_hue(df, x, hue, hue_str, norm=False, n_bins=50, figsize=(8,5), filename="", alpha=0.6, **kwargs):
    ''' plot distribution of given variable for each group.
    Args:
        df: Pandas dataframe
        x: variable to plot
        hue: additional category. seperate distribution will be plotted for each hue within the same plot.
        hue_str: dictionary to map hue value and name. i.e. 0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Return: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # filter group
    n_hue = df[hue].nunique()
    # loop over hue
    for j, hue_name in enumerate(sorted(df[hue].unique())):
        df_hue = df.loc[df[hue] == hue_name]
        df_hue[x].hist(bins=n_bins, histtype='step', ax=ax, label=hue_str[hue_name], density=norm, linewidth=1.5, **kwargs)
    # customize plot
    ax.set_xlabel(x)
    ax.set_ylabel("n")
    ax.grid(False)
    ax.legend()
    # customize and save plot
    plt.tight_layout()
    plt.savefig('plots/dist_%s.png' % filename)
    plt.cla()

def plot_line_groupby(df, x, y, groupby, group_label, ylog=False, x_label="", y_label="", figsize=(20,6), filename=""):
    ''' create line plot for different group in the same axes.
    Args:
        df: Pandas dataframe
        x: column used for x
        y: column to plot
        groupby: column representing different groups
        group_label: dictionary that maps gruop value to title. ex. {0:"AG1", 1:"AG2", etc.}
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax=ax.flatten()
    line = itertools.cycle(lines) 
    for name, df_group in df.groupby(groupby):
        if x=="index":
            df_group[y].plot(kind='line', legend=True, label=group_label[name], linewidth=2.0, linestyle=next(line))
        else:
            df_group.set_index(x)[y].plot(kind='line', legend=True, label=group_label[name], linewidth=2.0, linestyle=next(line))
    # customize and save plot
    if ylog:
        ax[0].set_yscale('log')
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    #ax[0].grid(False)
    plt.tight_layout()
    plt.savefig('plots/line_%s.png' % filename)
    plt.cla()

def plot_line_multiple_cols(df, x, list_y, legends, x_label, y_label, ylog=False, figsize=(20,6), filename=""):
    ''' create line plot from multiple columns in the same axes.
    Args:
        df: Pandas dataframe
        x: column used for x
        list_y: list of column names to plot
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax=ax.flatten()
    line = itertools.cycle(lines) 
    for i, y in enumerate(list_y):
        if x=="index":
            df[y].plot(kind='line', linewidth=2.0, label=legends[i], linestyle=next(line))
        else:
            df.set_index(x)[y].plot(kind='line', linewidth=2.0, label=legends[i], linestyle=next(line))
    # customize and save plot
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    if ylog:
        ax[0].set_yscale('log')
    #ax[0].grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/line_%s.png' % filename)
    plt.cla()
    return
    
def plot_heatmap(df, x_label, y_label, figsize=(20,6), filename="", cmap="Blues", **kwargs):
    ''' create heatmap from given dataframe
    Args:
        df: Pandas dataframe
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # plot heatmap
    ax = sns.heatmap(df, annot=True, cmap=cmap, **kwargs)
    # customize and save plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig('plots/heatmap_%s.png' % filename)


def plot_scatter(df, x, y, x_label="", y_label="", figsize=(20,6), filename=""):
    ''' create heatmap from given dataframe
    Args:
        df: Pandas dataframe
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # plot heatmap
    ax = df.plot(x=x,y=y)
    # customize and save plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig('plots/scatter_%s.png' % filename)
    plt.cla()
    return

def plot_box(df, x, y, title, x_label, y_label, ylim=None, figsize=(20,6), filename="", **kwargs):
    ''' create box plot from pandas dataframe.
    Args:
        df: Pandas dataframe
        x: column used for x axis
        y: column used for y axis
        others: plotting options
    Return:
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
        plt.savefig('plots/box_%s.png' % filename)
    return


def plot_heatmap_group(df_list, n_subplot_columns, x_label, y_label, df_err_list=None, group_map=None, ylim=None, figsize=(20,6), filename="", cmap="Blues", **kwargs):
    ''' create heatmap from given dataframe
    Args:
        df: list of dataframe
        x: column used for x axis
        y: column used for y axis
        n_subplot_columns: number of columns of subplot grid
        df_err_list: dataframe presenting uncertainty (optional)
    	group: column used to separate plots into multiple subplots
        group_map: dictionary that maps values in group column to string. ex. {10:"Sector1", etc.}
        others: plotting options
    Return:
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
            df_annot = df_list[group_name].applymap(lambda x: '%.3f' % float(x))\
                            + df_err_list[group_name].applymap(lambda x: ' (%.3f)' % float(x))
        else:
            df_annot = True

        #sns.heatmap(df_list[group_name], ax=ax[i], annot=True, cmap=cmap, **kwargs)
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
        plt.savefig('plots/heatmap_group_%s.png' % filename)
    return

