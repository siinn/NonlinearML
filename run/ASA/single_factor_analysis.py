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
input_path = '/mnt/mainblob/nonlinearML/data/ASA/xlsx/ASA_G2_data.r4.xlsx'
plot_path = 'output/ASA/EDA/'

# Set available features and labels
features = ['PM', 'DIFF']
labels = ['Residual']

# Set train and test period
test_begin = "2011-01-01"
test_end = "2018-01-01"

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
    df = pd.read_excel(input_path)
    df = df.drop(['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'], axis=1)

    # Read return data and append to data
    df_return = pd.read_excel(
        '/mnt/mainblob/nonlinearML/data/ASA/xlsx/ASA_G2_data.r2.append.xlsx')
    df['fmRet'] = df_return['fmRet']

    # Convert time into datetime format
    df[time] = df[time].apply(
        utils.datenum_to_datetime,
        matlab_origin=729025,
        date_origin=datetime(1996,1,1))

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, time, test_begin, test_end)

    # Only test train or test
    df = df_test

    #-----------------------------------------------------------------------
    # Heaetmap of average return
    #-----------------------------------------------------------------------
    """ Average return as a functin of two features."""
    for n_classes in [3, 5, 10]:
        # Discretize two features
        df = utils.discretize_variables_by_month(
            df=df, variables=features,
            n_classes=n_classes, suffix="n%s" %n_classes,
            class_names=\
                ["Low 1"]\
                +[str(x+1) for x in range(1,n_classes-1)]\
                +["High %s" %n_classes], month=time)

        # Calculate average return by each bin
        features_disc = [x+"_n%s" %n_classes for x in features]
        df_return_mean = df.groupby(features_disc).mean()[month_return]\
                           .unstack(1).transpose().sort_index(ascending=False)

        # Calculate average edge by each bin
        features_disc = [x+"_n%s" %n_classes for x in features]
        df_edge_mean = df.groupby(features_disc).mean()['Edge']\
                         .unstack(1).transpose().sort_index(ascending=False)

        # Calculate average adjusted edge by each bin
        features_disc = [x+"_n%s" %n_classes for x in features]
        df_edge_adj_mean = df.groupby(features_disc).mean()['Edge_Adj']\
                         .unstack(1).transpose().sort_index(ascending=False)

        # Calculate average residual by each bin
        features_disc = [x+"_n%s" %n_classes for x in features]
        df_residual_mean = df.groupby(features_disc).mean()['Residual']\
                         .unstack(1).transpose().sort_index(ascending=False)


        # Get min and max
        vmin = min(
            df_return_mean.min().min(),
            df_edge_mean.min().min(),
            df_edge_adj_mean.min().min())

        vmax = max(
            df_return_mean.max().max(),
            df_edge_mean.max().max(),
            df_edge_adj_mean.max().max())

        # plot heatmap of mean and standard deviation of return
        plot.plot_heatmap(
            df=df_return_mean,
            x_label=features[0], y_label=features[1], figsize=(8,6),
            annot_kws={'fontsize':10}, annot=True, fmt='.3f', cmap=cmap,
            vmin=vmin, vmax=vmax,
            filename=plot_path+"mean_%s_%s" % (month_return, n_classes))

        # plot heatmap of mean and standard deviation of return
        plot.plot_heatmap(
            df=df_edge_mean,
            x_label=features[0], y_label=features[1], figsize=(8,6),
            annot_kws={'fontsize':10}, annot=True, fmt='.3f', cmap=cmap,
            vmin=vmin, vmax=vmax,
            filename=plot_path+"mean_%s_%s" % ('Edge', n_classes))

        # plot heatmap of mean and standard deviation of return
        plot.plot_heatmap(
            df=df_edge_adj_mean,
            x_label=features[0], y_label=features[1], figsize=(8,6),
            annot_kws={'fontsize':10}, annot=True, fmt='.3f', cmap=cmap,
            vmin=vmin, vmax=vmax,
            filename=plot_path+"mean_%s_%s" % ('Edge_Adj', n_classes))

        # plot heatmap of mean and standard deviation of return
        plot.plot_heatmap(
            df=df_residual_mean,
            x_label=features[0], y_label=features[1], figsize=(8,6),
            annot_kws={'fontsize':10}, annot=True, fmt='.3f', cmap=cmap,
            vmin=vmin, vmax=vmax,
            filename=plot_path+"mean_%s_%s" % ('Residual', n_classes))


    print("Successfully completed all tasks")
