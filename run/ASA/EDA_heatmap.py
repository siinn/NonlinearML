#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import itertools
import math
import matplotlib
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.plot as plot
import NonlinearML.lib.utils as utils

#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------
# Set input and output path
input_path = '/mnt/mainblob/nonlinearML/data/ASA/csv/ASA_G2_data.r5.p1.csv'

# Set available features and labels
features = ['PM_Exp', 'Diff_Exp']
labels = ['Residual']

# Set train and test period
test_begin = "2011-01-01"
test_end = "2018-01-01"

# Set
time = 'smDate'

# colors map
cmap = matplotlib.cm.get_cmap('RdYlGn', 100)
colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
cmap=sns.color_palette(colors)


# Calculate standard error of mean or median
def getStandardError(x, confidence=1, median=False):
    """ Return standard error = confidence level * std / sqrt(N)"""
    scale = 1
    if median:
        # Scale factor for standard error of median
        scale=math.sqrt(math.pi/2)
    return scale*confidence*x.std() / (x.count()**(1/2))


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df_all = pd.read_csv(input_path, parse_dates=[time])

    # Calculate relative forward month return
    FM_REL_RETURN = 'fmRet'
    FQ_REL_RETURN = 'fqRelRet'

    # If True, FM return is standardized each month
    if False:
        df_all[FM_REL_RETURN] = df_all.groupby('smDate')\
                .apply(
                    lambda x:x['fmRet']- x['fmRet']\
                            .mean()).reset_index()['fmRet']

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df_all, time, test_begin, test_end)

    # Split training set into two period
    second_half_begin = "2001-01-01"
    second_half_end = "2018-01-01"
    df_train1, df_train2 = cv.train_test_split_by_date(
        df_train, time, second_half_begin, second_half_end)



    #---------------------------------------------------------------------------
    # Loop over different number of bins
    #---------------------------------------------------------------------------
    #for n_classes in [3, 5, 10]:
    for n_classes in [5]:

        #-----------------------------------------------------------------------
        # Heaetmap of average return
        #-----------------------------------------------------------------------
        """ Average return as a functin of two features."""
        datasets = {
                'All':df_all,
                '1996-2010':df_train,
                '1996-2010 excl. 2001 and 2009':df_train1,
                '2001 or 2009':df_train2,
                '2011-2017':df_test}

        # Set min and max of color bar
        vrange = {
                FM_REL_RETURN   :(-0.01, 0.01),
                FQ_REL_RETURN   :(-0.01, 0.01),
                'Total_Edge'    :(-0.01, 0.01),
                'Edge_Adj'      :(-0.01, 0.01),
                'Residual'      :(-0.01, 0.01),
                }

        # Name of discretized features
        features_disc = [x+"_n%s" %n_classes for x in features]

        for dataset in datasets:

            #-------------------------------------------------------------------
            # Set output path
            #-------------------------------------------------------------------
            plot_path = 'output/ASA/EDA/heatmap/%s/%s/' % (n_classes, dataset)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            # Set dataframe
            df = datasets[dataset]

            # Discretize two features
            df = utils.discretize_variables_by_month(
                df=df, variables=features,
                n_classes=n_classes, suffix="n%s" %n_classes,
                class_names=\
                    ["%02d (Low)" %n_classes]\
                    +["%02d" %x for x in range(n_classes-1,1,-1)]\
                    +["01 (High)"], month=time)

            # Change type from category to string so that we can add
            # additional row, column for average
            df[features_disc] = df[features_disc].astype(str)

            # For each z variable
            df_mean = {}
            df_std_mean = {}
            df_std_median = {}
            df_median = {}

            for z in [FM_REL_RETURN, FQ_REL_RETURN,
                    'Total_Edge', 'Edge_Adj', 'Residual']:
                #---------------------------------------------------------------
                # Heatmap of average
                #---------------------------------------------------------------
                # Calculate average value by each bin
                df_mean[z] = df.groupby(features_disc).mean()[z]\
                        .unstack(1).transpose().sort_index(ascending=False)


                # Add average to heatmap
                df_mean[z].loc['All'] = {x[0]:x[1] for x in
                        zip(df_mean[z].columns,
                            df.groupby(features_disc[0]).mean()[z])}
                df_mean[z]['All'] = df.groupby(features_disc[1]).mean()[z]
                df_mean[z].loc['All','All'] = df[z].mean()

                df_mean[z].sort_index(axis=0,inplace=True)
                df_mean[z].sort_index(axis=1, ascending=False, inplace=True)

                # Calculate standard error of mean
                df_std_mean[z] = df.groupby(features_disc)\
                              .apply(getStandardError)[z]\
                              .unstack(1).transpose()\
                              .sort_index(ascending=False)

                # Add std across rows and columns 
                df_std_mean[z].loc['All'] = {x[0]:x[1] for x in
                    zip(df_std_mean[z].columns,
                        df.groupby(features_disc[0])\
                          .apply(getStandardError)[z])}
                df_std_mean[z]['All'] = df.groupby(features_disc[1]).apply(getStandardError)[z]
                df_std_mean[z].loc['All','All'] = getStandardError(df[z])

                df_std_mean[z].sort_index(axis=0,inplace=True)
                df_std_mean[z].sort_index(axis=1, ascending=False, inplace=True)

                # Plot heatmap of mean
                plot.plot_heatmap(
                    df=df_mean[z],
                    annot=df_mean[z].applymap(lambda x:"%.3f" %x)\
                        + df_std_mean[z].applymap(lambda x:"\n(%.3f)" %x),
                    x_label=features[0], y_label=features[1], figsize=(8,6),
                    annot_kws={'fontsize':10}, fmt='s', cmap=cmap,
                    vmin=vrange[z][0], vmax=vrange[z][1],
                    filename=plot_path+"mean/%s_%s" % (z, n_classes))

                #---------------------------------------------------------------
                # Heatmap of median
                #---------------------------------------------------------------
                # Calculate median value by each bin
                df_median[z] = df.groupby(features_disc).median()[z]\
                                   .unstack(1).transpose()\
                                   .sort_index(ascending=False)

                # Add median to heatmap
                df_median[z].loc['All'] = {x[0]:x[1] for x in
                        zip(df_median[z].columns,
                            df.groupby(features_disc[0]).median()[z])}
                df_median[z]['All'] = df.groupby(features_disc[1]).median()[z]
                df_median[z].loc['All','All'] = df[z].median()

                df_median[z].sort_index(axis=0,inplace=True)
                df_median[z].sort_index(axis=1, ascending=False, inplace=True)

                # Calculate standard error of mean
                df_std_median[z] = df.groupby(features_disc)\
                              .apply(getStandardError, median=True)[z]\
                              .unstack(1).transpose()\
                              .sort_index(ascending=False)

                # Add std across rows and columns 
                df_std_median[z].loc['All'] = {x[0]:x[1] for x in
                    zip(df_std_median[z].columns,
                        df.groupby(features_disc[0])\
                          .apply(getStandardError, median=True)[z])}
                df_std_median[z]['All'] = df.groupby(features_disc[1])\
                                     .apply(getStandardError, median=True)[z]
                df_std_median[z].loc['All','All'] = getStandardError(df[z], median=True)

                df_std_median[z].sort_index(axis=0,inplace=True)
                df_std_median[z].sort_index(axis=1, ascending=False, inplace=True)


                # Plot heatmap of median
                plot.plot_heatmap(
                    df=df_median[z],
                    annot=df_median[z].applymap(lambda x:"%.3f" %x)\
                        + df_std_median[z].applymap(lambda x:"\n(%.3f)" %x),
                    x_label=features[0], y_label=features[1], figsize=(8,6),
                    annot_kws={'fontsize':10}, fmt='s', cmap=cmap,
                    vmin=vrange[z][0], vmax=vrange[z][1],
                    filename=plot_path+"median/%s_%s" % (z, n_classes))



                #---------------------------------------------------------------
                # Count number of samples in heapmap
                #---------------------------------------------------------------
                # Count number of samples per month per group
                df_count = df.groupby(features_disc+[time])\
                             .count()[features[0]].unstack().transpose()
                # Rename columns
                df_count.columns.set_levels(
                    ["PM %s" %x for x in range(1,n_classes+1)], level=0, inplace=True)
                df_count.columns.set_levels(
                    ["Diff %s" %x for x in range(1,n_classes+1)], level=1, inplace=True)

                N_BINS=50
                plot.plot_distribution(
                    df=df_count, columns=df_count.columns,
                    n_rows=n_classes, n_columns=n_classes, histtype='bar',
                    color='dodgerblue', edgecolor='black',
                    bins=[N_BINS]*len(df_count.columns),
                    xrange=[(0,N_BINS)]*len(df_count.columns),
                    ylog=[], ylim=[], xticks_minor=True,
                    title=[], x_label=['Samples']*len(df_count.columns),
                    y_label=[], figsize=(25,25),
                    filename=plot_path+"%s_%s_count" %(z, n_classes))


    print("Successfully completed all tasks")
