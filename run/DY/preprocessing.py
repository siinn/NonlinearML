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
input_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/DYPO_2020.01.30.csv'
output_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/DYPO_2020.01.30.p1.csv'
log_path = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/'
plot_path = 'output/DY/EDA/Prod_2020.01.30/'

# Path to training data for comparison
training_path = '../EnhancedDividend/data/Data_EM_extended.csv'

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
winsorize_alpha_lower = 0.02
winsorize_alpha_upper = 0.98
#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

#-------------------------------------------------------------------------------
# define functions
#-------------------------------------------------------------------------------
#def count_null(df, columns):
#    '''
#    Calculate fraction of null values in given columns.
#    Args:
#        df: Pandas dataframe
#        columns: columns of interest
#    Return:
#        p_null: fraction of null values in dictionary. ex. {"column1": 0.5, ...}
#    '''
#    p_null = {}
#    for column in columns:
#        p_null[column] = df[column].isnull().mean()
#    return p_null
#
#def winsorize_series(s):
#    '''Winsorize each var by month
#    Note: use pandas quantitle function instead of scipy.winsorize function,
#    because of Scipy winsorize function has NaN handling bug, as noted in previous text
#    Args: a Series
#    Return: winsorized series
#    '''
#    s = pd.Series(s)
#    q = s.quantile([winsorize_alpha_lower, winsorize_alpha_upper])
#    if isinstance(q, pd.Series) and len(q) == 2:
#        s[s < q.iloc[0]] = q.iloc[0]
#        s[s > q.iloc[1]] = q.iloc[1]    
#    return s
#
#
#def winsorize_df(df, features):
#    ''' Winsorize given features
#    Args:
#        df: Pandas dataframe
#        features: list of column names to winsorize
#    Return:
#        dataframe with the given columns winsorized
#    ''' 
#    for feature in features:
#        df[feature] = winsorize_series(df[feature])
#    return df
#
#def standardize_series(col):
#    '''Normalize each column by month mean and std'''
#    return (col - col.mean()) / col.std()
#
#def standardize_df(df, features):
#    '''Standardize dataframe by month mean and std'''
#    for feature in features:
#      df[feature] = standardize_series(df[feature])    
#    return df
#
#
#def remove_missing_targets(df, targets):
#    '''the observations are removed if target variables are missing.
#    The fraction of missing returns are printed.'''
#    # check null values
#    null_fraction = count_null(df, targets)
#    for i, key in enumerate(null_fraction):
#        io.message("Removing the observations with missing %s (%.4f)" % (targets[i], null_fraction[key]))
#    # remove null
#    return df.dropna(subset=targets)  
#
#def impute_data(df, method, features):
#    ''' Impute missing data using the given imputation method. 
#    Args:
#        df: dataframe
#        method: available options: month, securityId_ff, securityId_average
#        features: features to impute.
#    Return:
#        imputed dataframe
#    '''
#    if impute_method == "month":
#        df = impute_by_month(df, features)
#    elif impute_method == "securityId_ff":
#        df = impute_by_securityID(df, features)
#    elif impute_method == "securityId_average":
#        df = impute_by_securityID_forward(df, features)
#    else:
#        io.message("Impute method is not valid.")
#    return df
#
#def impute_by_month(df, features):
#    '''Impute missing data with the mean Z score within the same month group'''
#    df[features] = df[features].fillna(0, inplace=False)
#    return df
#
#
#def datenum_to_datetime(x, matlab_origin, date_origin):
#    """ Convert matlab timestamp to Timestamp."""
#    return date_origin + relativedelta(days=(x-matlab_origin))


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
    # Winsorize MSCIEM at 2% and 98% and get median and std
    #---------------------------------------------------------------------------
    # Print
    io.title("Raw data")
    print_statistics(
        dfs=[df, df.loc[df['IsMSCIEM']==1], df.loc[df['IsMSCIEM']==0]],
        label=["All", "MSCIEM", "Rest"], features=features)

    # Winsorize MSCIEM data
    df_MSCIEM = df.loc[df['IsMSCIEM']==1]
    df_MSCIEM = winsorize_df(
        df_MSCIEM, features,
        winsorize_alpha_lower, winsorize_alpha_upper)
    df.iloc[:len(df_MSCIEM)] = df_MSCIEM

    # Get median and std from MSCIEM
    stats = {}
    for feat in features:
        stats[feat] = [df_MSCIEM[feat].median(), df_MSCIEM[feat].std()]

    #---------------------------------------------------------------------------
    # Standardize using median std obtained from MSCIEM
    #---------------------------------------------------------------------------
    for feat in features:
        df[feat] = (df[feat] - stats[feat][0]) / stats[feat][1]

    # Print
    io.title("After standardization")
    print_statistics(
        dfs=[df, df.loc[df['IsMSCIEM']==1], df.loc[df['IsMSCIEM']==0]],
        label=["All", "MSCIEM", "Rest"], features=features)

    #---------------------------------------------------------------------------
    # Winsorize all data at -3 and +3
    #---------------------------------------------------------------------------
    def truncate(x, minval, maxval):
        """ truncate value at (minval, maxval)"""
        if x > 3:
            return 3
        elif x < -3:
            return -2
        return x
    for feat in features:
        df[feat] = df[feat].apply(lambda x: truncate(x,-3,3))

    # Print
    io.title("After winsorization")
    print_statistics(
        dfs=[df, df.loc[df['IsMSCIEM']==1], df.loc[df['IsMSCIEM']==0]],
        label=["All", "MSCIEM", "Rest"], features=features)

    #---------------------------------------------------------------------------
    # Impute missing data
    #---------------------------------------------------------------------------
    if impute_data:
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
    # Rename features
    #---------------------------------------------------------------------------
    io.title("Rename features")
    io.message("Original name: %s" %str(features))
    io.message("New name: %s" %str(rename_features))
    if rename_features:
        df = df.rename(
            {feature:new_feature for feature, new_feature \
                in zip(features, rename_features)}, axis=1)

    #---------------------------------------------------------------------------
    # Comparing distribution with training data
    #---------------------------------------------------------------------------
    def stack_df(df, label):
        """ Temporary helper function to stack dataframe"""
        df = df[rename_features]\
            .stack()\
            .reset_index()\
            .rename({'level_1':'feature', 0:'Value'},axis=1)\
            .drop('level_0', axis=1)
        df['Dataset']=label
        return df
    # Load training data
    df_train = pd.read_csv(training_path, index_col=None, parse_dates=[time])
    # Winsorize monthly
    for feat in rename_features:
        df_train[feat] = df_train[feat].apply(lambda x: truncate(x,-3,3))
    df_last_train = df_train\
        .loc[df_train['smDate']==df_train['smDate'].unique()[-2]]
    df_MSCIEM = df.loc[df['IsMSCIEM']==1]
    df_nonMSCIEM = df.loc[df['IsMSCIEM']==0]
    # Concat dataframes for plotting
    df_compare = pd.concat([
        stack_df(df_train, 'Train (1997/01-2019/04)'),
        stack_df(df_last_train, 'Train (2019/04)'),
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
                (df_year['smDate'].iloc[0].year, df_year['smDate'].iloc[0].month
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

