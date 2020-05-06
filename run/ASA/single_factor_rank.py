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
feature = 'PM_Exp'
#feature = 'Diff_Exp'
#feature = 'Total_Edge'
#feature = 'Edge_Adj'

# Label
month_return = "fmRet"

# Set train and test period
train_begin = "1996-01-01"
train_end = "2010-12-31"
test_begin = "2011-01-01"
test_end = "2017-11-01"

# Set output path
plot_path = 'output/ASA/single_factor/rank_transformd/%s/' % feature

# Set number of classes
n_classes = 5

# Set
date_column = 'smDate'

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

    # Discretize feature
    df = utils.discretize_variables_by_month(
        df=df, variables=[feature],
        n_classes=n_classes, suffix="n%s" %n_classes,
        class_names=[x for x in range(n_classes,0,-1)], month=date_column)

    # Split dataset into train and test dataset
    df_train, df_test = cv.train_test_split_by_date(
        df, date_column, test_begin, test_end,
        train_begin, train_end)


    #-----------------------------------------------------------------------
    # Calculate return
    #-----------------------------------------------------------------------

    # Calculate cumulative return using trained model
    df_backtest_train, df_backtest_test = backtest.perform_backtest(
            pred_train=df_train, pred_test=df_test,
            col_pred='%s_n%s' % (feature, n_classes),
            list_class=[x+1 for x in range(n_classes)],
            label_fm=month_return, time=date_column)

    # Calculate difference in cumulative return, annual return, and IR
    df_diff_test = backtest.calculate_diff_IR(
        df=df_backtest_test,
        top=1, bottom=5,
        class_label={x+1:str(int(x+1)) for x in range(n_classes)},
        class_reg=month_return,
        time=date_column,
        col_pred='%s_n%s' % (feature, n_classes))

    df_diff_train = backtest.calculate_diff_IR(
        df=df_backtest_train,
        top=1, bottom=5,
        class_label={x+1:str(int(x+1)) for x in range(n_classes)},
        class_reg=month_return,
        time=date_column,
        col_pred='%s_n%s' % (feature, n_classes))

    # Make cumulative return plot
    grouplabel = {1:'1 (High %s)' %feature, 2:'2', 3:'3', 4:'4', 5:'5 (Low %s)' %feature}
    plot_backtest.plot_cumulative_return(
        df_backtest_train, df_backtest_test, month_return,
        #group_label={x+1:str(int(x+1)) for x in range(n_classes)},
        group_label=grouplabel,
        figsize=(12,8),
        filename=plot_path+"return_by_group",
        date_column=date_column,
        train_ylim=(-1,5),
        test_ylim=(-1,3),
        col_pred='%s_n%s' % (feature, n_classes))

    # Make difference in cumulative return plot
    plot_backtest.plot_cumulative_return_diff(
	list_cum_returns=[df_backtest_train],
	top=1, bottom=5,
	class_label={1:'1 (High %s)' %feature, 5:'5 (Low %s)' %feature},
	list_labels=['Rank'], label_reg=feature,
	figsize=(12,8),
	date_column=date_column,
	filename=plot_path+"return_diff_train",
	ylim=(-0.5,5),
        col_pred='%s_n%s' % (feature, n_classes))

    plot_backtest.plot_cumulative_return_diff(
	list_cum_returns=[df_backtest_test],
	top=1, bottom=5,
	class_label={1:'1 (High %s)' %feature, 5:'5 (Low %s)' %feature},
	list_labels=['Rank'], label_reg=feature,
	figsize=(12,8),
	date_column=date_column,
	filename=plot_path+"return_diff_test",
	ylim=(-0.5, 5),
        col_pred='%s_n%s' % (feature, n_classes))




    #-----------------------------------------------------------------------
    # Save output
    #-----------------------------------------------------------------------
    df_train.to_csv(plot_path+"df_train.csv")
    df_test.to_csv(plot_path+"df_test.csv")

    df_diff_train.to_csv(plot_path+"df_diff_train.csv")
    df_diff_test.to_csv(plot_path+"df_diff_test.csv")
    df_backtest_test.to_csv(plot_path+"df_backtest_test.csv")
    df_backtest_train.to_csv(plot_path+"df_backtest_train.csv")

    # Pivot tables and save as excel
    df_backtest_test\
        .pivot(index=date_column, columns='%s_n%s' % (feature, n_classes))\
        .to_excel(plot_path+"df_backtest_test.xlsx")

    df_backtest_train\
        .pivot(index=date_column, columns='%s_n%s' % (feature, n_classes))\
        .to_excel(plot_path+"df_backtest_train.xlsx")



    print("Successfully completed all tasks")
