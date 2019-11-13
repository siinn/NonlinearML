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
INPUT_PATH = '/mnt/mainblob/earnings-transcripts-sp/data_meta_feature/return_meta_feature_R1000_ALL_v5.csv'
plot_path = 'output/earnings_transcripts/EDA/'

# Set available features and labels
features = ['ceo_sentiment', 'other_exe_sentiment']
labels = ['Fwd1MTotalReturnHedgedUSD']

# Set train and test period
test_begin = "2018-01-01"
test_end = "2018-12-31"

# Set
time = 'RebalDate'

# return label
month_return = 'Fwd1MTotalReturnHedgedUSD'

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
    df = pd.read_csv(INPUT_PATH, index_col=None, parse_dates=[time])

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, time, test_begin, test_end)

    #-----------------------------------------------------------------------
    # Heaetmap of average return (Train)
    #-----------------------------------------------------------------------
    """ Average return as a functin of two features."""
    for n_classes in [3, 5, 10]:
        # Discretize two features
        df_train = utils.discretize_variables_by_month(
            df=df_train, variables=features,
            n_classes=n_classes, suffix="n%s" %n_classes,
            class_names=\
                ["Low 0"]\
                +[str(x) for x in range(1,n_classes-1)]\
                +["High %s" %n_classes], month=time)

        df_test = utils.discretize_variables_by_month(
            df=df_test, variables=features,
            n_classes=n_classes, suffix="n%s" %n_classes,
            class_names=\
                ["Low 0"]\
                +[str(x) for x in range(1,n_classes-1)]\
                +["High %s" %n_classes], month=time)

        # Calculate average return by each bin
        features_disc = [x+"_n%s" %n_classes for x in features]
        df_train_return_mean = df_train.groupby(features_disc).mean()[month_return]\
                           .unstack(1).transpose().sort_index(ascending=False)

        # Calculate average return by each bin
        features_disc = [x+"_n%s" %n_classes for x in features]
        df_test_return_mean = df_test.groupby(features_disc).mean()[month_return]\
                           .unstack(1).transpose().sort_index(ascending=False)

        # Get min and max
        vmin = min(
            df_train_return_mean.min().min(),
            df_test_return_mean.min().min())

        vmax = max(
            df_train_return_mean.max().max(),
            df_test_return_mean.max().max())

        # plot heatmap of mean and standard deviation of return
        plot.plot_heatmap(
            df=df_train_return_mean,
            x_label=features[0], y_label=features[1], figsize=(8,6),
            annot_kws={'fontsize':10}, annot=True, fmt='.3f', cmap=cmap,
            vmin=vmin, vmax=vmax,
            filename=plot_path+"train_mean_%s_%s_%s_%s" % \
                (month_return, features[0], features[1], n_classes))

        # plot heatmap of mean and standard deviation of return
        plot.plot_heatmap(
            df=df_test_return_mean,
            x_label=features[0], y_label=features[1], figsize=(8,6),
            annot_kws={'fontsize':10}, annot=True, fmt='.3f', cmap=cmap,
            vmin=vmin, vmax=vmax,
            filename=plot_path+"test_mean_%s_%s_%s_%s" % \
                (month_return, features[0], features[1], n_classes))



    print("Successfully completed all tasks")
