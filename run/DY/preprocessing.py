#!/usr/bin/env python
# import common python libraries
from __future__ import division
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
import numpy as np
import os
import pandas as pd
from matplotlib.dates import DateFormatter
import seaborn as sns
import warnings


# Import custom libraries
from NonlinearML.plot.plot import *
from NonlinearML.lib.utils import *
from NonlinearML.lib.preprocessing import *
import NonlinearML.lib.io as io

# Supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None

#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------

# Set input and output path
output_tag = 'p3'
input_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/raw/Data_EM_extended.r3.csv'
output_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/preprocessed/Data_EM_extended.%s.csv' % output_tag
log_path = 'output/DY/EDA/preprocessed/'
plot_path = 'output/DY/EDA/preprocessed/'

# Select algorithm to run    
run_eda                 = True
impute_data             = True
save_output             = False

# Set available features and labels
#features = ['DividendYield','EG','Payout_E','DY_dmed','PO_dmed','EG_dmed']
features = ['DividendYield','Payout_E','DY_dmed','PO_dmed']
col_drop = []

# Set
time = 'smDate'

# Set winsorization alpha
winsorize_alpha_lower = 0.02
winsorize_alpha_upper = 0.98
#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

#-------------------------------------------------------------------------------
# define plotting functions
#-------------------------------------------------------------------------------
def plot_null(df, columns, figsize=(15,5), filename=""):
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
    pdf.plot.bar(y=0, ax=ax, legend=False, color="black", alpha=0.5)
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

def plot_null_vs_time(df, time, columns, n_rows=4, n_columns=4, figsize=(20,12), xticks_interval=20, filename="", ylim=(0,1)):
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
        # plot fraction of null values
        ax[i].plot(df_null.index, df_null[column])
        # customize axes
        ax[i].xaxis.set_major_formatter(DateFormatter("%Y"))
        ax[i].set_xlabel(column)
        ax[i].set_ylabel("Missing data (%)")
        ax[i].set_ylim(ylim)
    # remove extra subplots
    #for x in np.arange(len(columns),len(ax),1):
    #    fig.delaxes(ax[x])
    plt.tight_layout()
    #fig.autofmt_xdate()
    plt.savefig('%s.png' %filename)
    plt.cla()
    return

def print_statistics(dfs, label, features):
    """ Print median, mean, and std of multiple DataFrame
    Args:
        dfs: list of DataFrames
        label: list of names corresponding to dataframes
        features: list of columns of interest
    Return:
        None
    """
    for df, name in zip(dfs, label):
        io.message(name)
        for feat in features:
            io.message(
                "\t>%s: mean=%.2f, median = %.2f, std = %.2f, min = %.2f, max = %.2f"\
                %(
                    feat, df[feat].mean(), df[feat].median(), df[feat].std(),
                    df[feat].min(), df[feat].max()))
    return


if __name__ == "__main__":

    #---------------------------------------------------------------------------
    # Load input data
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=None, parse_dates=[time])

    # Set logging configuration
    io.setConfig(path=log_path, filename="log.txt")

    #---------------------------------------------------------------------------
    # Perform EDA of raw data
    #---------------------------------------------------------------------------
    if run_eda:
        io.title("Plotting data distribution")
        # Set columns to plot
        columns = [x for x in df.columns.to_list() if x != time]
        n_plots=len(columns)
        n_columns = 4
        n_rows = math.ceil(n_plots/n_columns)

        # Make plot (linear)
        plot_distribution(
            df, columns=columns, n_rows=n_rows, n_columns=n_columns, 
            bins=[50]*n_plots, ylog=[False]*n_plots,
            xrange=[], ylim=[], title=[""]*n_plots,
            x_label=[], y_label=["Samples"]*n_plots, figsize=(20,6),
            filename=plot_path+"raw_dist_linear", color='royalblue')

        # Make plot (log)
        plot_distribution(
            df, columns=columns, n_rows=n_rows, n_columns=n_columns, 
            bins=[50]*n_plots, ylog=[True]*n_plots,
            xrange=[], ylim=[], title=[""]*n_plots,
            x_label=[], y_label=["Samples"]*n_plots, figsize=(20,6),
            filename=plot_path+"raw_dist_log", color='crimson')


        # Plot percentage of null values
        plot_null(
            df, features, figsize=(20,6), filename=plot_path+"raw_null_fraction")
    
        # plot fraction of null values as a function of time
        plot_null_vs_time(
            df, time=time, columns=df.columns,
            n_rows=n_rows, n_columns=n_columns, figsize=(20, 6),
            filename=plot_path+"raw_null_fraction_time", ylim=(0,0.01))


    #---------------------------------------------------------------------------
    # Impute missing data
    #---------------------------------------------------------------------------
    if impute_data:
        io.title("Imputing missing values")
        # Replace missing values with 0 (monthly mean)
        for col in features:
            df[col] = df[col].apply(lambda x:0 if pd.isnull(x) else x)

        # Drop samples with no return (< 0.003% of total data)
        df = df.dropna(subset=col_drop)

    #---------------------------------------------------------------------------
    # Plot distribution after preprocessing
    #---------------------------------------------------------------------------
    if run_eda:

        # Set columns to plot
        columns = [x for x in df.columns.to_list() if x != time]
        n_plots=len(columns)
        n_columns = 4
        n_rows = math.ceil(n_plots/n_columns)

        # Make plot (linear)
        plot_distribution(
            df, columns=columns, n_rows=n_rows, n_columns=n_columns, 
            bins=[50]*n_plots, ylog=[False]*n_plots,
            xrange=[], ylim=[], title=[""]*n_plots,
            x_label=[], y_label=["Samples"]*n_plots, figsize=(20,6),
            filename=plot_path+"processed_dist_linear", color='royalblue')

        # Make plot (log)
        plot_distribution(
            df, columns=columns, n_rows=n_rows, n_columns=n_columns, 
            bins=[50]*n_plots, ylog=[True]*n_plots,
            xrange=[], ylim=[], title=[""]*n_plots,
            x_label=[], y_label=["Samples"]*n_plots, figsize=(20,6),
            filename=plot_path+"processeddist_log", color='crimson')


        # Plot percentage of null values
        plot_null(
            df, features, figsize=(20,6), filename=plot_path+"processed_null_fraction")
    
        # plot fraction of null values as a function of time
        plot_null_vs_time(
            df, time=time, columns=df.columns,
            n_rows=n_rows, n_columns=n_columns, figsize=(20, 6),
            filename=plot_path+"processed_null_fraction_time", ylim=(0,0.01))

    #---------------------------------------------------------------------------
    # Save as csv
    #---------------------------------------------------------------------------
    if save_output:
        df.to_csv(output_path, index=False)

    
    io.message("Successfully completed all tasks!")

