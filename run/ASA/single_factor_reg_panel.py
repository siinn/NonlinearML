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
import statsmodels.api as sm

# Import custom libraries
import NonlinearML.lib.backtest as backtest
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.io as io
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.plot as plot
import NonlinearML.lib.utils as utils

#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '../data/ASA/csv/ASA_G2_data.r5.p1.csv'

# Set feature of interest
#feature = 'PM_Exp'
#feature = 'Diff_Exp'
#feature = 'Edge_Adj'
feature = 'Total_Edge'

# Label
month_return = "fmRet"
quater_return = "fqRelRet"
target = 'Residual'
#target = 'fqRelRet'

# Set train and test period
train_begin = None
train_end = None
test_begin = "2011-01-01"
test_end = "2018-01-01"

# Set output path
output_path = 'output/ASA/single_factor/reg/%s/' % feature

# Set number of classes
n_classes = 5

# Set top and bottom class for calculating return
inverse_relation = True

# Set time column
date_column = 'smDate'

# colors map
cmap = matplotlib.cm.get_cmap('RdYlGn', n_classes)
colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
cmap=sns.color_palette(colors)

#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == "__main__":

    # Set logging configuration
    io.setConfig(path=output_path, filename="log.txt")
    io.title('Running single factor regression:')
    io.message("Feature = %s" % feature)
    io.message("Target = %s" % target)

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(INPUT_PATH, index_col=None, parse_dates=[date_column])

    #---------------------------------------------------------------------------
    # Run pool regression
    #---------------------------------------------------------------------------
    io.title("Running panel regression on each month:")
    month_index, res = [], []

    for month in sorted([month for month in df[date_column].unique()]):
        mask = df[date_column]==month
        # Fit linear regression on each month
        lm = sm.OLS(
            endog=df.loc[mask][target].values,
            exog=df.loc[mask][feature].values.reshape(-1,1))
        results = lm.fit()
        # Append results
        month_index.append(month)
        # MacKinnon and White’s (1985) heteroskedasticity robust standard errors
        conf = 2 # Error bars will represent 2 sigma
        res.append([results.params[0], conf*results.HC3_se[0]])

    df_panel = pd.DataFrame(data=res, index=pd.DatetimeIndex(month_index))\
        .rename({0:"Coeff",1:"StdErr"}, axis=1)

    #---------------------------------------------------------------------------
    # Make plots
    #---------------------------------------------------------------------------
    io.title("Creating plots of coefficients")

    # Plot coefficients as a function of time
    plot.plot_errorbar(
        df=df_panel, x='index', y='Coeff', error='StdErr',
        x_label="Time", y_label="Coefficient", figsize=(14,4),
        filename=output_path+"coeff_%s" %feature, legend=False,
        fmt='o', markeredgecolor='black', markerfacecolor='dodgerblue',
        elinewidth=1, ecolor='gray')

    #---------------------------------------------------------------------------
    # Calculate statistics
    #---------------------------------------------------------------------------
    io.title("panel regression statistics")
    # Separate train and test
    df_panel_train = df_panel.loc[df_panel.index < test_begin].copy()
    df_panel_test = df_panel.loc[df_panel.index >= test_begin].copy()
    
    # Print statistics
    io.message("Train begin = %s, end=%s"
        % (df_panel_train.index[0].strftime("%Y-%m"),
        df_panel_train.index[-1].strftime("%Y-%m")))
    io.message("\tMedian = %.4f, Mean = %.4f (+/-%.4f)"
        % (
        df_panel_train['Coeff'].median(),
        df_panel_train['Coeff'].mean(),
        df_panel_train['StdErr'].std()))

    io.message("Test begin = %s, end=%s"
        % (df_panel_test.index[0].strftime("%Y-%m"),
        df_panel_test.index[-1].strftime("%Y-%m")))
    io.message("\tMedian = %.4f, Mean = %.4f (+/-%.4f)"
        % (
        df_panel_test['Coeff'].median(),
        df_panel_test['Coeff'].mean(),
        df_panel_test['StdErr'].std()))


    io.message("Successfully completed all tasks!")
