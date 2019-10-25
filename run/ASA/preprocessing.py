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

# Supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None

#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------

# Set input and output path
input_path = '/mnt/mainblob/nonlinearML/data/ASA/xlsx/ASA_G2_data.r2.xlsx'
output_path = '/mnt/mainblob/nonlinearML/data/ASA/csv/ASA_G2_data.r2.p1.csv'
plot_path = 'output/EDA/ASA/'

# Select algorithm to run    
run_eda                 = True
impute_data             = True
save_output             = True

# Set available features and labels
features = ['PM', 'DIFF']
labels = ['Residual']

# Set
time = 'smDate'

# Set winsorization alpha
winsorize_alpha_lower = 0.01
winsorize_alpha_upper = 0.99

#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

#-------------------------------------------------------------------------------
# define functions
#-------------------------------------------------------------------------------
def count_null(df, columns):
    '''
    Calculate fraction of null values in given columns.
    Args:
        df: Pandas dataframe
        columns: columns of interest
    Return:
        p_null: fraction of null values in dictionary. ex. {"column1": 0.5, ...}
    '''
    p_null = {}
    for column in columns:
        p_null[column] = df[column].isnull().mean()
    return p_null

def winsorize_series(s):
    '''Winsorize each var by month
    Note: use pandas quantitle function instead of scipy.winsorize function,
    because of Scipy winsorize function has NaN handling bug, as noted in previous text
    Args: a Series
    Return: winsorized series
    '''
    s = pd.Series(s)
    q = s.quantile([winsorize_alpha_lower, winsorize_alpha_upper])
    if isinstance(q, pd.Series) and len(q) == 2:
        s[s < q.iloc[0]] = q.iloc[0]
        s[s > q.iloc[1]] = q.iloc[1]    
    return s


def winsorize_df(df, features):
    ''' Winsorize given features
    Args:
        df: Pandas dataframe
        features: list of column names to winsorize
    Return:
        dataframe with the given columns winsorized
    ''' 
    for feature in features:
        df[feature] = winsorize_series(df[feature])
    return df

def standardize_series(col):
    '''Normalize each column by month mean and std'''
    return (col - col.mean()) / col.std()

def standardize_df(df, features):
    '''Standardize dataframe by month mean and std'''
    for feature in features:
      df[feature] = standardize_series(df[feature])    
    return df


def remove_missing_targets(df, targets):
    '''the observations are removed if target variables are missing.
    The fraction of missing returns are printed.'''
    # check null values
    null_fraction = count_null(df, targets)
    for i, key in enumerate(null_fraction):
        print("Removing the observations with missing %s (%.4f)" % (targets[i], null_fraction[key]))
    # remove null
    return df.dropna(subset=targets)  

def impute_data(df, method, features):
    ''' Impute missing data using the given imputation method. 
    Args:
        df: dataframe
        method: available options: month, securityId_ff, securityId_average
        features: features to impute.
    Return:
        imputed dataframe
    '''
    if impute_method == "month":
        df = impute_by_month(df, features)
    elif impute_method == "securityId_ff":
        df = impute_by_securityID(df, features)
    elif impute_method == "securityId_average":
        df = impute_by_securityID_forward(df, features)
    else:
        print("Impute method is not valid.")
    return df

def impute_by_month(df, features):
    '''Impute missing data with the mean Z score within the same month group'''
    df[features] = df[features].fillna(0, inplace=False)
    return df


def datenum_to_datetime(x, matlab_origin, date_origin):
    """ Convert matlab timestamp to Timestamp."""
    return date_origin + relativedelta(days=(x-matlab_origin))


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


if __name__ == "__main__":

    #---------------------------------------------------------------------------
    # Load input data
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_excel(input_path)
    df = df.drop(['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'], axis=1)

    # Read return data and append to data
    df_return = pd.read_excel(
        '/mnt/mainblob/nonlinearML/data/ASA/xlsx/ASA_G2_data.r2.append.xlsx')
    df['fmRet'] = df_return['fmRet']

    # Convert time into datetime format
    df[time] = df[time].apply(
        datenum_to_datetime,
        matlab_origin=729025,
        date_origin=datetime(1996,1,1))


    #---------------------------------------------------------------------------
    # Perform EDA using raw data
    #---------------------------------------------------------------------------
    if run_eda:

        # Plot feature and return distribution (linear)
        n_plots=10
        plot_distribution(
            df, columns=[x for x in df.columns.to_list() if x != time],
            n_rows=4, n_columns=3, 
            bins=[100]*n_plots, ylog=[False]*n_plots,
            xrange=[], ylim=[], title=[""]*n_plots,
            x_label=[], y_label=["Samples"]*n_plots, figsize=(16,12),
            filename=plot_path+"dist_linear")

        plot_distribution(
            df, columns=[x for x in df.columns.to_list() if x != time],
            n_rows=4, n_columns=3, 
            bins=[100]*n_plots, ylog=[True]*n_plots,
            xrange=[], ylim=[], title=[""]*n_plots,
            x_label=[], y_label=["Samples"]*n_plots, figsize=(16,12),
            filename=plot_path+"dist_log")


        # Plot percentage of null values
        plot_null(
            df, features, figsize=(15,8), filename=plot_path+"null_fraction")
    
        # plot fraction of null values as a function of time
        n_rows=4
        n_columns=3
        plot_null_vs_time(
            df, time=time, columns=df.columns,
            n_rows=n_rows, n_columns=n_columns,
            figsize=(20,20), filename=plot_path+"null_fraction_time", ylim=(0,0.01))

    #---------------------------------------------------------------------------
    # Impute missing data
    #---------------------------------------------------------------------------
    if impute_data:
        # Replace missing values with 0 (monthly mean)
        for col in ['PM', 'DIFF']:
            df[col] = df[col].apply(lambda x:0 if pd.isnull(x) else x)

        # Drop samples with no return (< 0.003% of total data)
        df = df.dropna(subset=['Residual'])


    #---------------------------------------------------------------------------
    # Save as csv
    #---------------------------------------------------------------------------
    if save_output:
        df.to_csv(output_path, index=False)

    
    print("Successfully completed all tasks!")

