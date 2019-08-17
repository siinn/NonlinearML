import numpy as np
import pandas as pd

from NonlinearML.plot.plot import *
from NonlinearML.lib.preprocessing import count_null
from NonlinearML.plot.style import load_matplotlib
plt = load_matplotlib()


#----------------------------------------------
# define plotting functions
#----------------------------------------------
def plot_null(df, columns, figsize=(15,5), filename="", **kwargs):
    '''
    Plot percentage of null values for each of given columns.
    Args:
        df: Pandas dataframe
        columns: columns of interest
        others: plotting optoins
    Return: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # get fraction of null values
    p_null = count_null(df, df.columns)
    pdf = pd.DataFrame.from_dict(p_null, orient='index')
    # make bar plot
    pdf.plot.bar(y=0, ax=ax, legend=False, **kwargs)
    # annotate numbers
    y_offset = 0.01
    for p in ax.patches:
        bar = p.get_bbox()
        val = "{:+.3f}".format(bar.y1 + bar.y0)        
        ax.annotate(val, (bar.x0, bar.y1 + y_offset))
    # customize plot and save
    ax.set_ylabel("Fraction of null values")
    ax.set_ylim(0,0.3)
    plt.tight_layout()
    plt.savefig('%s.png' %filename)
    return

def plot_null_vs_time(df, time, columns, n_rows=4, n_columns=4, figsize=(20,12), xticks_interval=20, filename=""):
    '''
    Plots fraction of null data as a function of time for each column.
    Args:
        df: Pandas dataframe
        time: time column
        columns: columns of interest
        others: plotting options
    Return: None
    '''
    # calculate percecntage of valid data for each month
    df_null = df.groupby(time).apply(lambda x: x.isnull().mean())\
                  .sort_index()\
                  .drop([time], axis=1)
    # create figure and axes
    fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize)
    ax = ax.flatten()
    # count number of non-null data for each feature grouped by month
    columns = [x for x in columns if x != time] # remove time column
    for i, column in enumerate(columns):
        # fraction of null YOY changes
        ax[i].plot(df_null.index, df_null[column])
        # customize axes
        ax[i].set_xlabel(column)
        ax[i].set_ylabel("Missing data (%)")
        ax[i].set_ylim(0,1)
        #ax[i].set_xlim(x1, x2)
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(90)
        # Set axis frequency
        ax[i].set_xticks(ax[i].get_xticks()[::xticks_interval])
    # remove extra subplots
    for x in np.arange(len(columns),len(ax),1):
        fig.delaxes(ax[x])
    plt.tight_layout()
    plt.savefig('%s.png' %filename)
    plt.cla()
    return
