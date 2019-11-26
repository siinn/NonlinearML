#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import itertools
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
plot_path = 'output/ASA/EDA/correlation/'

# Set available features and labels
features = ['PM_Exp', 'Diff_Exp']
feature1 = features[0]
feature2 = features[1]
labels = ['Residual']

# Set
time = 'smDate'

# return label
month_return = "fmRet"

# colors map
cmap = matplotlib.cm.get_cmap('RdYlGn', 10)
colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
cmap=sns.color_palette(colors)

#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

if __name__ == "__main__":

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path)

    #-----------------------------------------------------------------------
    # 
    #-----------------------------------------------------------------------
    """ Average return as a functin of two features."""
    for n_classes in [3, 5, 10]:
    #for n_classes in [10]:
        # Discretize two features
        df = utils.discretize_variables_by_month(
            df=df, variables=features,
            n_classes=n_classes, suffix="n%s" %n_classes,
            class_names=\
                ["Low 1"]\
                +[str(x+1) for x in range(1,n_classes-1)]\
                +["High %s" %n_classes], month=time)

        #-------------------------------------------------------------------
        # Calculate top - bottom of average return calculated each month
        #-------------------------------------------------------------------
        def diff_top_bot(df, feature):
            """ Calculate difference between and top and bottom"""
            top = df.loc[\
                df[feature+"_n%s" %n_classes]=='High %s' %n_classes]\
                [feature+'_avg'].values
            bottom = df.loc[\
                df[feature+'_n%s' %n_classes]=='Low 1']\
                [feature+'_avg'].values
            return pd.Series(top - bottom)
        # Feature 1 (ex. PM)
        diff_feature1 = df.groupby([time, feature1+"_n%s" %n_classes]).mean()\
            .reset_index()\
            .rename({feature1:feature1+'_avg'}, axis=1)\
            .groupby(time)\
            .apply(diff_top_bot, feature=feature1)\
            .rename({0:'diff_%s_avg' %feature1}, axis=1)

        # Feature 2 (ex. DIFF)
        diff_feature2 = df.groupby([time, feature2+"_n%s" %n_classes]).mean()\
            .reset_index()\
            .rename({feature2:feature2+'_avg'}, axis=1)\
            .groupby(time)\
            .apply(diff_top_bot, feature=feature2)\
            .rename({0:'diff_%s_avg' %feature2}, axis=1)

        # Concatenate two dataframe
        diff_feature1["diff_%s_avg" % feature2] = diff_feature2
        df_corr = diff_feature1 # alias

        print(df_corr.corr())

        # Make scatter plot
        plot.plot_scatter(
            df_corr,
            x="diff_%s_avg" % feature1,
            y="diff_%s_avg" % feature2,
            x_label="Difference in average return\nTop - bottom by %s (monthly)" % feature1,
            y_label="Difference in average return\nTop - bottom by %s (monthly)" % feature2,
            figsize=(10,10),
            filename=plot_path+"corr_%s_%s_%s" % (feature1, feature2, n_classes),
            legend=False,
            leg_loc='center left', legend_title=None, bbox_to_anchor=(1, 0.5),
            ylim=False, xlim=False, color='gray', s=100)

        


    print("Successfully completed all tasks")
