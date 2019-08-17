#!/usr/bin/env python
# import common python libraries
from __future__ import division
import numpy as np
import math
import numpy as np
import os
import pandas as pd

# Import custom libraries
import NonlinearML.lib.utils as utils
import NonlinearML.lib.preprocessing as prep
import NonlinearML.plot.eda as eda
import NonlinearML.plot.plot as plot




#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------

# Set input and output path
#input_path = \
#'/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.csv'
input_path = '../data/\
Data_for_AssetGrowth_Context.r5.p2.csv'
output_path = \
'/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r5.p2.csv'
plot_path = 'plots/EDA/'
# Set True for development
debug = True

# Select algorithm to run    
run_eda                 = True
run_preprocessing       = True
examine_processed_data  = True
save_results            = False

# Set imputation method. available options:
#   month, securityId_ff, securityId_average
impute_method           =  "month"

# Set available features and labels
features = [
    'GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG',
    'SG', 'CVROIC', 'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum']
labels = ["fmTotalReturn", "fqTotalReturn"]

# Set
time = 'eom'

# Set winsorization alpha
winsorize_alpha_lower = 0.01
winsorize_alpha_upper = 0.99

#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
utils.create_folder(plot_path)


if __name__ == "__main__":

    #---------------------------------------------------------------------------
    # Load input data
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, index_col=[0], parse_dates=[time])

    # Convert time into datetime format
    #df[time] = df[time].apply(utils.to_datetime, date_format='%Y%m')

    #---------------------------------------------------------------------------
    # Add negative features
    #---------------------------------------------------------------------------
    neg_features_to_add = ['AG', 'FCFA']
    neg_features = ['-'+x for x in neg_features_to_add]
    df[neg_features] = df[neg_features_to_add].transform(lambda x:-x)

    # Add negative features to the original list
    features = features + neg_features

    # Get Number of features, columns, and rows for plotting
    n_features = len(features)
    n_columns = 4
    n_rows = math.ceil(n_features/n_columns)

    #---------------------------------------------------------------------------
    # Perform EDA using raw data
    #---------------------------------------------------------------------------
    if run_eda:

        # Plot feature distribution (linear)
        plot.plot_distribution(
            df, columns=features, n_rows=n_rows, n_columns=n_columns, 
            bins=[50]*len(features), ylog=[False]*len(features), xrange=[],
            ylim=[], title=[""]*len(features),
            x_label=[], y_label=["Samples"]*len(features), figsize=(20,18),
            filename=plot_path+"dist_features_linear")

        # Plot feature distribution (log)
        plot.plot_distribution(
            df, columns=features, n_rows=n_rows, n_columns=n_columns, 
            bins=[50]*len(features), ylog=[True]*len(features), xrange=[],
            ylim=[], title=[""]*len(features),
            x_label=[], y_label=["Samples"]*len(features), figsize=(20,18),
            filename=plot_path+"dist_features_log")

        # Plot percentage of null values
        eda.plot_null(
            df, features, figsize=(15,8), filename=plot_path+"null_fraction")
            #color="red", edgecolor='b')
    
        # plot fraction of null values as a function of time
        eda.plot_null_vs_time(
            df, time="eom", columns=df.columns,
            n_rows=math.ceil(len(df.columns)/n_columns), n_columns=n_columns,
            figsize=(20,20), filename=plot_path+"null_fraction_time")


    #---------------------------------------------------------------------------
    # Run preprocessing
    #---------------------------------------------------------------------------
    if run_preprocessing:

        print("Running preprocessing..")

        # Apply winsorization. Not applied for AG 
        print(" > winsorizing data with (%s, %s)" \
            % (winsorize_alpha_lower, winsorize_alpha_upper))
        df = prep.winsorize_df(
            df=df,
            features=[
                x for x in features 
                if x not in ["GS", 'GICSSubIndustryNumber']],
            lower=winsorize_alpha_lower,
            upper=winsorize_alpha_upper)

        # Standardize features
        print(" > applying standardization data")
        df = prep.standardize_df(
            df=df,
            features=[
                x for x in features
                if x not in ["GS", 'GICSSubIndustryNumber']])

        # Remove the observations that are missing target variables
        print(" > removing the observations with missing target data")
        df = prep.remove_missing_targets(df, targets=labels)

        # Fill empty GICSSubIndustryNumber with 99999999, then keep only 
        # first 2 digits of GICSSubIndustryNumber
        print(" > truncating GICS number")
        df = df.fillna({"GICSSubIndustryNumber":99999999})
        df["GICSSubIndustryNumber"] = df["GICSSubIndustryNumber"]\
            .apply(lambda x: float(str(x)[:2]))

        # Assign quintile and tertile classes to return
        df = utils.discretize_variables_by_month(
            df=df, variables=[
                'fmTotalReturn', 'fqTotalReturn'], # 0 is high
            labels_tertile={'fmTotalReturn':[2,1,0], 'fqTotalReturn':[2,1,0]},
            labels_quintile={
                'fmTotalReturn':[4,3,2,1,0],
                'fqTotalReturn':[4,3,2,1,0]}) # 0 is high
        # impute missing values
        print(" > Imputing missing data")
        df = prep.impute_data(df, impute_method, features)

    #---------------------------------------------------------------------------
    # Examine processed data
    #---------------------------------------------------------------------------
    if examine_processed_data:

        # Plot feature distribution (linear)
        plot.plot_distribution(
            df, columns=features, n_rows=n_rows, n_columns=n_columns,
            color='red', alpha=0.5, bins=[50]*n_features,
            ylog=[False]*n_features, xrange=[], ylim=[], title=[""]*n_features,
            x_label=[], y_label=["Samples"]*n_features, figsize=(20,18),
            filename=plot_path+"dist_features_proc_linear")

        # Plot feature distribution (log)
        plot.plot_distribution(
            df, columns=features, n_rows=n_rows, n_columns=n_columns,
            color='red', alpha=0.5, bins=[50]*n_features,
            ylog=[True]*n_features, xrange=[], ylim=[], title=[""]*n_features,
            x_label=[], y_label=["Samples"]*n_features, figsize=(20,18),
            filename=plot_path+"dist_features_proc_log")
    

    #---------------------------------------------------------------------------
    # Save results
    #---------------------------------------------------------------------------
    if save_results:
      df.to_csv(output_path, header=True)

    
    print("Successfully completed all tasks!")

