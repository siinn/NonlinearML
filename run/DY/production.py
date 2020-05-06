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
input_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/Production.20200305.csv'
output_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/Production.20200305.p1.csv'
log_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/'
plot_path = 'output/DY/EDA/Prod_2020.03.05/'

# Path to training data for comparison
training_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/raw/Data_EM_extended.r3.csv'

# Select algorithm to run    
run_eda                 = True
impute_data             = False
save_output             = True

# Set available features and labels
#features = ['DividendYield','EG','Payout_E','DY_dmed','PO_dmed','EG_dmed']
features = ['DY', 'PO']
rename_features = ['DY_dmed', 'PO_dmed']
col_drop = []

# Set
time = 'smDate'

# Set winsorization alpha
winsorize_alpha_lower = 0.03
winsorize_alpha_upper = 0.97
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
    #df = pd.read_csv(input_path, index_col=None, parse_dates=[time])
    df = pd.read_csv(input_path, index_col=None)
    df[time] = datetime(datetime.today().year, datetime.today().month, 1)

    # Set logging configuration
    df_MSCIEM = df.loc[df['IsMSCIEM']==1]
    df_nonMSCIEM = df.loc[df['IsMSCIEM']==0]
    # Concat dataframes for plotting
    df_compare = pd.concat([
        stack_df(df_train, 'Train (1997/01-2019/12)'),
        stack_df(df_MSCIEM, 'MSCIEM'),
        stack_df(df_nonMSCIEM, 'non-MSCIEM')])

    plot_dist_groupby_hue(
        df=df_compare, x='Value', y_label='Samples', group_var='feature',
        group_title={col:col for col in df_compare['feature'].unique()},
        hue='Dataset',
        hue_str={col:col for col in df_compare['Dataset'].unique()},
        norm=True, x_range=(-4, 4),
        n_subplot_columns=2, n_bins=100, figsize=(16,8),
        filename=plot_path+"dist_comparison",
        linewidth=2, histtype='step')

    #---------------------------------------------------------------------------
    # Evolution of training data
    #---------------------------------------------------------------------------
    # Load training data
    df_train = pd.read_csv(training_path, index_col=None, parse_dates=[time])
    # Winsorize monthly
    #for feat in rename_features:
    #    df_train[feat] = df_train[feat].apply(lambda x: truncate(x,-3,3))
    df_by_year = []
    for year in range(1998,2019,2):
        start = datetime.strptime(str(year)+"-01", "%Y-%m")
        end = datetime.strptime(str(year+2)+"-12", "%Y-%m")
        df_year = df_train\
            .loc[(df_train['smDate']>=start) & (df_train['smDate']<=end)]
        df_by_year.append(stack_df(df_year, '%s/%s-%s/%s' %\
                (df_year['smDate'].iloc[0].year, df_year['smDate'].iloc[0].month,
                df_year['smDate'].iloc[-1].year, df_year['smDate'].iloc[-1].month)))
    # Concat dataframes for plotting
    df_compare = pd.concat(df_by_year)
    # Make plot
    plot_dist_groupby_hue(
        df=df_compare, x='Value', y_label='Samples', group_var='feature',
        group_title={col:col for col in df_compare['feature'].unique()},
        hue='Dataset',
        hue_str={col:col for col in df_compare['Dataset'].unique()},
        norm=True, x_range=(-4, 4),
        n_subplot_columns=2, n_bins=100, figsize=(16,8),
        filename=plot_path+"dist_comparison_by_year",
        linewidth=2, histtype='step')



    #---------------------------------------------------------------------------
    # Save as csv
    #---------------------------------------------------------------------------
    if save_output:
        df.to_csv(output_path, index=False)

    
    io.message("Successfully completed all tasks!")

