#!/usr/bin/env python
# import common python libraries
from __future__ import division
import numpy as np
import math
import numpy as np
import os
import pandas as pd
import seaborn as sns


# Import custom libraries
from Asset_growth.lib.plots import *
from Asset_growth.lib.utils import *

#----------------------------------------------
# Set user options
#----------------------------------------------

# Set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.csv'
output_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.p2.csv'
plot_path = 'plots/EDA/'
# Set True for development
debug = True

# Select algorithm to run    
run_eda                 = True
run_preprocessing       = True
examine_processed_data  = True
save_results            = True

# Set imputation method. available options: month, securityId_ff, securityId_average
impute_method           =  "month"

# Set available features and labels
features = ['GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC', 'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum']
labels = ["fmTotalReturn", "fqTotalReturn"]

# Set
time = 'eom'

# Set winsorization alpha
winsorize_alpha_lower = 0.01
winsorize_alpha_upper = 0.99

#----------------------------------------------
# Create output folder
#----------------------------------------------
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

#----------------------------------------------
# define functions
#----------------------------------------------
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


#----------------------------------------------
# define plotting functions
#----------------------------------------------
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


if __name__ == "__main__":

    #----------------------------------------------
    # Load input data
    #----------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=[0], parse_dates=[time])

    # Convert time into datetime format
    df[time] = df[time].apply(to_datetime, date_format='%Y%m')

    #----------------------------------------------
    # Add negative features
    #----------------------------------------------
    neg_features_to_add = ['AG', 'FCFA']
    neg_features = ['-'+x for x in neg_features_to_add]
    df[neg_features] = df[neg_features_to_add].transform(lambda x:-x)

    # Add negative features to the original list
    features = features + neg_features

    # Get Number of features, columns, and rows for plotting
    n_features = len(features)
    n_columns = 4
    n_rows = math.ceil(n_features/n_columns)

    #----------------------------------------------
    # Perform EDA using raw data
    #----------------------------------------------
    if run_eda:


        # Plot feature distribution (linear)
        plot_distribution(df, columns=features, n_rows=n_rows, n_columns=n_columns, 
                          bins=[50]*len(features), ylog=[False]*len(features), xrange=[], ylim=[], title=[""]*len(features),
                          x_label=[], y_label=["Samples"]*len(features), figsize=(20,18), filename=plot_path+"dist_features_linear")

        # Plot feature distribution (log)
        plot_distribution(df, columns=features, n_rows=n_rows, n_columns=n_columns, 
                          bins=[50]*len(features), ylog=[True]*len(features), xrange=[], ylim=[], title=[""]*len(features),
                          x_label=[], y_label=["Samples"]*len(features), figsize=(20,18), filename=plot_path+"dist_features_log")

        # Plot percentage of null values
        plot_null(df, features, figsize=(15,8), filename=plot_path+"null_fraction")
    
        # plot fraction of null values as a function of time
        plot_null_vs_time(df, time="eom", columns=df.columns, n_rows=math.ceil(len(df.columns)/n_columns), n_columns=n_columns,
                          figsize=(20,20), filename=plot_path+"null_fraction_time")


    #----------------------------------------------
    # Run preprocessing
    #----------------------------------------------
    if run_preprocessing:

        print("Running preprocessing..")

        # Apply winsorization. Not applied for AG 
        print(" > winsorizing data with (%s, %s)" % (winsorize_alpha_lower, winsorize_alpha_upper))
        df = winsorize_df(df=df, features=[x for x in features if x not in ["GS", 'GICSSubIndustryNumber']])

        # Standardize features
        print(" > applying standardization data")
        df = standardize_df(df=df, features=[x for x in features if x not in ["GS", 'GICSSubIndustryNumber']])

        # Remove the observations that are missing target variables
        print(" > removing the observations with missing target data")
        df = remove_missing_targets(df, targets=labels)

        # Fill empty GICSSubIndustryNumber with 99999999, then keep only first 2 digits of GICSSubIndustryNumber
        print(" > truncating GICS number")
        df = df.fillna({"GICSSubIndustryNumber":99999999})
        df["GICSSubIndustryNumber"] = df["GICSSubIndustryNumber"].apply(lambda x: float(str(x)[:2]))

        # Assign quintile and tertile classes to return
        df = discretize_variables_by_month(df=df, variables=['fmTotalReturn', 'fqTotalReturn'],
                                           #labels_tertile={'fmTotalReturn':['T3', 'T2', 'T1'], 'fqTotalReturn':['T3', 'T2', 'T1']},
                                           #labels_quintile={'fmTotalReturn':['Q5', 'Q4', 'Q3', 'Q2', 'Q1'], 'fqTotalReturn':['Q5', 'Q4', 'Q3', 'Q2', 'Q1']})
                                           labels_tertile={'fmTotalReturn':[2,1,0], 'fqTotalReturn':[2,1,0]}, # 0 is high
                                           labels_quintile={'fmTotalReturn':[4,3,2,1,0], 'fqTotalReturn':[4,3,2,1,0]}) # 0 is high
        # impute missing values
        print(" > Imputing missing data")
        df = impute_data(df, impute_method, features)

    #----------------------------------------------
    # Examine processed data
    #----------------------------------------------
    if examine_processed_data:

        # Plot feature distribution (linear)
        plot_distribution(df, columns=features, n_rows=n_rows, n_columns=n_columns, color='red', alpha=0.5,
                          bins=[50]*n_features, ylog=[False]*n_features, xrange=[], ylim=[], title=[""]*n_features,
                          x_label=[], y_label=["Samples"]*n_features, figsize=(20,18), filename=plot_path+"dist_features_proc_linear")

        # Plot feature distribution (log)
        plot_distribution(df, columns=features, n_rows=n_rows, n_columns=n_columns, color='red', alpha=0.5,
                          bins=[50]*n_features, ylog=[True]*n_features, xrange=[], ylim=[], title=[""]*n_features,
                          x_label=[], y_label=["Samples"]*n_features, figsize=(20,18), filename=plot_path+"dist_features_proc_log")
    

    #----------------------------------------------
    # save results
    #----------------------------------------------
    if save_results:
      df.to_csv(output_path, header=True)

    
    print("Successfully completed all tasks!")

