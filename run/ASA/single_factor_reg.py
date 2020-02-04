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
from sklearn.linear_model import LinearRegression

# Import custom libraries
import NonlinearML.lib.backtest as backtest
import NonlinearML.lib.cross_validation as cv
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
feature = 'Diff_Exp'

# Label
month_return = "fmRet"
quater_return = "fqRelRet"

# Set train and test period
train_begin = None
train_end = None
test_begin = "2011-01-01"
test_end = "2018-01-01"

# Set output path
plot_path = 'output/ASA/single_factor/reg/%s/' % feature

# Set number of classes
n_classes = 5

# Set
date_column = 'smDate'

# colors map
cmap = matplotlib.cm.get_cmap('RdYlGn', n_classes)
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
    df = pd.read_csv(INPUT_PATH, index_col=None, parse_dates=[date_column])

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end,
        train_begin, train_end)

    # Fit linear regression on train and predict on train and test
    lm = LinearRegression(fit_intercept=False).fit(
                X=df_train[feature].values.reshape(-1,1),
                y=df_train[quater_return])
    df_train['pred'] = lm.predict(df_train[feature].values.reshape(-1,1))
    df_test['pred'] = lm.predict(df_test[feature].values.reshape(-1,1))
    print("Coefficient:")
    print("Feature = %s: %s" % (feature, lm.coef_))


    # Discretize prediction
    df_train = utils.discretize_variables_by_month(
        df=df_train, variables=['pred'],
        n_classes=n_classes, suffix="n%s" %n_classes,
        class_names=[x+1 for x in range(n_classes)], month=date_column)
    df_test = utils.discretize_variables_by_month(
        df=df_test, variables=['pred'],
        n_classes=n_classes, suffix="n%s" %n_classes,
        class_names=[x+1 for x in range(n_classes)], month=date_column)

    # Set top and bottom class
    if lm.coef_ > 0:
        top, bottom = n_classes, 1
    else:
        top, bottom = 1, n_classes


    #-----------------------------------------------------------------------
    # Calculate return
    #-----------------------------------------------------------------------

    # Calculate cumulative return using trained model
    df_backtest_train, df_backtest_test = backtest.perform_backtest(
            pred_train=df_train, pred_test=df_test,
            col_pred='pred_n%s' % n_classes,
            list_class=[x+1 for x in range(n_classes)],
            label_fm=month_return, time=date_column)

    # Calculate difference in cumulative return, annual return, and IR
    df_diff = backtest.calculate_diff_IR(
        df=df_backtest_test,
        top=top, bottom=bottom,
        class_label={x+1:str(int(x+1)) for x in range(n_classes)},
        class_reg=month_return,
        time=date_column,
        col_pred='pred_n%s' % n_classes)

    # Make cumulative return plot
    plot_backtest.plot_cumulative_return(
        df_backtest_train, df_backtest_test, month_return,
        group_label={x+1:str(int(x+1)) for x in range(n_classes)},
        figsize=(12,8),
        filename=plot_path+"return_by_group",
        date_column=date_column,
        train_ylim=(-0.5,3),
        test_ylim=(-0.5,3),
        col_pred='pred_n%s' % n_classes)


    #-----------------------------------------------------------------------
    # Save output
    #-----------------------------------------------------------------------
    df_diff.to_excel(plot_path+"df_diffs.xlsx")
    df_backtest_test.to_excel(plot_path+"df_backtest_test.xlsx")


    print("Successfully completed all tasks")
