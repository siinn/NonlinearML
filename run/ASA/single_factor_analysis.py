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
                ["Low 0"]\
                +[str(x) for x in range(1,n_classes-1)]\
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






#    #---------------------------------------------------------------------------
#    # AG, FCFA, sector
#    #---------------------------------------------------------------------------
#
#    if run_ag_fc == True:
#
#        # plot return by AG and FCFA
#        plot.plot_box(df=df, x=var_tertile, y=total_return, title="", color="white", linewidth=1, showmeans=True, ylim=(-0.5,2),
#                 hue="FCFA_discrete", #ylim=(0.7, 1.3),
#                 x_label="AG", y_label=total_return, figsize=(10,6), filename="./output/singlefactor/%s_by_AG_FCFA" %total_return)
#
#
#        # plot heatmap of number of samples within FCFA and return tertile groups for given AG group
#        for i_ag in df[var_tertile].unique():
#            plot.plot_heatmap(df=df.loc[df[var_tertile]==i_ag].groupby([return_tertile, "FCFA_discrete"]).count().iloc[:,0].unstack(level=-1),\
#                         x_label="fq total return", y_label="FCFA",\
#                         figsize=(10,7), filename="./output/singlefactor/%s_FCFA_AG%s" % (total_return, i_ag), fmt='.0f')
#
#
#        # plot average return of AG and FCFA tertile group
#        plot.plot_heatmap(df=df.groupby(["FCFA_discrete", var_tertile]).mean()[total_return].unstack(1).sort_index(ascending=False), square=True,\
#            annot_kws={'fontsize':20},
#            x_label="Asset growth", y_label="Free Cash Flow to Asset", annot=True,\
#            figsize=(8,6), filename="./output/singlefactor/%s_FCFA_AG_tertile" % (total_return), fmt='.3f')
#
#
#        # plot standard deviation of return for AG and FCFA tertile group by sector
#        df_std_list = {}
#        for i_industry in df[categories[0]].unique():
#            df_std_list[i_industry]=df.loc[df[categories[0]]==i_industry]\
#                                          .groupby([var_tertile, "FCFA_discrete"])\
#                                          .std()[total_return].unstack(1)
#        plot.plot_heatmap_group(df_list=df_std_list, n_subplot_columns=4,
#                           x_label="FCFA", y_label="AG", group_map=sector_map, figsize=(25,20),
#                           filename="./output/singlefactor/%s_std_FCFA_AG_sector_tertile" % (total_return), fmt='.3f', cmap=sns.light_palette("gray"))
#
#        # plot average return of AG and FCFA tertile group by sector
#        df_mean_list = {}
#        for i_industry in df[categories[0]].unique():
#            df_mean_list[i_industry]=df.loc[df[categories[0]]==i_industry]\
#                                          .groupby([var_tertile, "FCFA_discrete"])\
#                                          .mean()[total_return].unstack(1)
#        plot.plot_heatmap_group(df_list=df_mean_list, df_err_list=df_std_list, n_subplot_columns=4,
#                           x_label="FCFA", y_label="AG", group_map=sector_map, figsize=(25,20), fmt="s", cmap=sns.color_palette("RdBu_r", 7),
#                           filename="./output/singlefactor/%s_mean_FCFA_AG_sector_tertile" % (total_return))
#
#
#
#
#    print("Successfully completed all tasks")
