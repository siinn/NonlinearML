import numpy as np
import pandas as pd

import NonlinearML.lib.io as io

def _impute_missing_average(df, list_class, col_class, month_return, time):
    """ check if average return available. If not,
    set average return to 0. """
    for date in sorted(df[time].unique())[1:]:
        # Retrieve current return and previous asset
        df_curr = df.loc[df[time]==date].sort_values(col_class) 
        for classes in list_class:
            if classes not in df_curr[col_class].unique():
                io.message("\tFound a period in which mean return is not"\
                    + "available: date=%s, classes=%s" %(date, classes))
                df_add = df_curr.head(1).copy()
                df_add[month_return]=0
                df_add[col_class]=classes
                df = pd.concat([df, df_add], axis=0, ignore_index=True)\
                    .sort_values(col_class)
    return df.set_index(col_class)

def calculate_return(
    df, list_class, col_class, month_return, time="eom"):
    """ Calculate cumulative return of each classes (ex. AG1-AG5)
    Args:
        df: Pandas dataframe
        list_class: List of classes. Ex. [0, 1, 2]
        col_class: Ex. "fqReturn_tercile" or "pred"
        month_return: monthly return used to calculate cumulative return
        time: time column
    Return:
        df_avg: dataframe representing cumulative return at each time, t.
    """
    # Calculate average return of each classes
    df_avg = df.groupby([time,col_class])[month_return].mean()\
        .reset_index()
    # Find the starting month of cumulative return
    first_month = sorted(df_avg[time].unique())[0]
    cumulative_begin_month = np.datetime64(first_month, 'M') - 1
    # Add zero return as the beginnig of cumulative return
    for classes in df[col_class].unique():
        df_avg = df_avg.append(
            {
                time:cumulative_begin_month,
                col_class:classes, month_return:0.0},
            ignore_index=True)
    # Create cumulative return column
    df_avg["cumulative_asset"] = 1.0
    df_avg["cumulative_return"] = 0.0
    df_avg["annual_return"] = 0.0
    # If average return is not available, set average return to 0.
    df_avg = _impute_missing_average(
        df=df_avg, col_class=col_class, list_class=list_class,
        month_return=month_return, time=time)
    # loop over each date
    for t, date in enumerate(sorted(df_avg[time].unique())[1:]):
        # get last month
        prev_month = np.datetime64(date, 'M') - 1
        # retrieve current and previous month dataframe
        df_curr = df_avg.loc[df_avg[time]==date]
        df_prev = df_avg.loc[df_avg[time]==prev_month]
        # Retrieve Cumulative asset  
        cum_asset = df_prev["cumulative_asset"] 
        # Calculate cumulative and annual return
        """ cumulative asset = (r_1 + 1)* ... *(r_t + 1)
            cumulative return = cumulative asset - 1
            annualized return = [cumulative asset]^(12/months) - 1"""
        df_avg.loc[df_avg[time]==date, "cumulative_asset"] = \
            cum_asset*(1+df_curr[month_return])
        df_avg.loc[df_avg[time]==date, "cumulative_return"] = \
            cum_asset*(1+df_curr[month_return]) - 1
        df_avg.loc[df_avg[time]==date, "annual_return"] = \
            pow(cum_asset*(1+df_curr[month_return]), (12/(t+1))) - 1
    return df_avg.reset_index()



def calculate_diff_return(df_cum_return, top, bottom, class_label, output_col,
    time="eom", col_pred="pred"):
    ''' Calculate difference in return between top and bottom classes.
        Example. Q1 - Q3 or D1 - D10 where 1 is high return.
    Args:
        df_cum_return: cumulative returns calculated by
            "calculate_return".
        top, bottom: class name of top and bottom class
        class_label: dictionary that maps class name to label
        output_col: name of column representing the difference in return
        time: name of time column
    Return:
        df_diff: dataframe containing month, difference in return and
            "col_pred" label for plotting
    '''
    # Set time as index
    df_cum_return = df_cum_return.set_index(time)
    # Calculate difference
    io.message("Calculating difference in return: top (%s) - bottom (%s)." \
        % (str(class_label[top]), str(class_label[bottom])))

    df_diff = pd.DataFrame(
        df_cum_return.loc[
            df_cum_return[col_pred]==top]["cumulative_return"]\
        - df_cum_return.loc[
            df_cum_return[col_pred]==bottom]["cumulative_return"])
    # Assign "Q1+Q2-Q3" as pred value for plotting
    df_diff[col_pred] = output_col
    return df_diff

def calculate_diff_IR(
    df, top, bottom, class_label, class_reg, time, col_pred="pred"):
    ''' Calculate difference in IR between top and bottom classes.
        Example. Q1 - Q3 or D1 - D10 where 1 is high return.
    Args:
        df: cumulative returns calculated by function "calculate_return"
        top, bottom: class name of top and bottom class
        class_label: dictionary that maps class name to label
        time: name of time column
    Return:
        df_diff: dataframe containing month, difference in return and
            "pred" label for plotting
    '''
    io.message("Calculating difference in IR: top (%s) - bottom (%s)." \
        % (str(class_label[top]), str(class_label[bottom])))
    # Set time as index
    df = df.set_index(time)
    # Calculate difference
    df_top = df.loc[df[col_pred]==top]
    df_bot = df.loc[df[col_pred]==bottom]
    # Calculate top - bottom
    df_diff = df_top - df_bot
    df_diff = df_diff.rename({col:col+"_diff" for col in df_diff.columns}, axis=1)
    # Sort and assign index representing time t in months.
    df_diff = df_diff.sort_values(time).reset_index()
    # Calculate std of difference in returns
    std = []
    for i in df_diff.index.values:
        std.append(
                df_diff[class_reg+"_diff"].iloc[:int(i+1)]\
                    .std()*pow(12,(1/2)))
    # Append std of difference in returns to dataframe
    df_diff['std_diff'] = std
    # Calculate IR
    df_diff['IR'] = df_diff.apply(
        lambda x: x['annual_return_diff'] / x['std_diff'], axis=1)
    return df_diff[[
        time, 'cumulative_return_diff', 'annual_return_diff',
        'std_diff', 'IR']]




def perform_backtest(pred_train, pred_test, list_class, label_fm, time, col_pred="pred"):
    ''' Wrapper of calculate_return function.
    Calculate cumulative return using the prediction.
    Args:
        pred_train, pred_test: prediction of train and test dataset. This is
            an output of utils.prediction function.
        list_class: list of classes. Ex. [0, 1, 2]
        label_fm: monthly return used for calculating cumulative return
        time: column name representing time
        col_class: column name representing predicted class
    Return:
        df_backtest_train: cumulative return calculated from train dataset
        df_backtest_test: cumulative return calculated from test dataset
        model: trained model
    '''
    # Calculate cumulative return
    io.title("Calculating cumulative return")
    io.message(" > Calculating cumulative return of train dataset..")
    df_backtest_train = calculate_return(
        df=pred_train, list_class=list_class,
        col_class=col_pred,
        month_return=label_fm, time=time)
    io.message(" > Calculating cumulative return of train dataset..")
    df_backtest_test = calculate_return(
        df=pred_test, 
        col_class=col_pred, list_class=list_class,
        month_return=label_fm, time=time)
    return df_backtest_train, df_backtest_test


