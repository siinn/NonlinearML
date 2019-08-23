import numpy as np
import pandas as pd

import NonlinearML.lib.io as io



#def cumulative_return(df, var_classes, total_return, time="eom"):
#    ''' Calculate cumulative return of each classes (ex. AG1-AG5)
#    Args:
#        df: Pandas dataframe
#        var_classes: column name representing classes of the variable of interest (ex. AG_quintile)
#        total_return: return column
#        time: time column
#    Return:
#        df_avg: Pandas dataframe representing cumulative return for each unit time
#    '''
#    def _impute_missing_average(df, var_classes, total_return, time):
#        ''' check if average return available. If not, set average return to 0. '''
#        for date in sorted(df[time].unique())[1:]:
#            df_curr = df.loc[df[time]==date].sort_values(var_classes) # retrieve current return and previous asset
#            for classes in df[var_classes].unique():
#                if classes not in df_curr[var_classes].unique():
#                    io.message("Found a period in which mean return is not available: date=%s, classes=%s" %(date, classes))
#                    df_add = df_curr.head(1).copy()
#                    df_add[total_return]=0
#                    df_add[var_classes]=classes
#                    df = pd.concat([df, df_add], axis=0, ignore_index=True).sort_values(var_classes)
#        return df.set_index(var_classes)
#    # calculate average return of each classes of the variable of interest (ex.AG)
#    df_avg = df.groupby([time,var_classes])[total_return].mean().reset_index()
#    # find the starting month of cumulative return
#    first_month = sorted(df_avg[time].unique())[0]
#    cumulative_begin_month = np.datetime64(first_month, 'M') - 1
#    # add zero return as the beginnig of cumulative return
#    for classes in df[var_classes].unique():
#        df_avg = df_avg.append({time:cumulative_begin_month, var_classes:classes, total_return:0.0}, ignore_index=True)
#    # create cumulative return column
#    df_avg["cumulative_asset"] = 1.0
#    df_avg["cumulative_return"] = 0.0
#    # if average return is not available, set average return to 0.
#    df_avg = _impute_missing_average(df=df_avg, var_classes=var_classes, total_return=total_return, time=time)
#    # loop over each date
#    for date in sorted(df_avg[time].unique())[1:]:
#        # get last month
#        prev_month = np.datetime64(date, 'M') - 1
#        # retrieve current and previous month dataframe
#        df_curr = df_avg.loc[df_avg[time]==date]
#        df_prev = df_avg.loc[df_avg[time]==prev_month]
#        # calculate cumulative asset and return
#        df_avg.loc[df_avg[time]==date, "cumulative_asset"] = df_prev["cumulative_asset"] * (1 + df_curr[total_return])
#        df_avg.loc[df_avg[time]==date, "cumulative_return"] = df_prev["cumulative_asset"] * (1 + df_curr[total_return]) - 1
#    return df_avg.reset_index()



def cumulative_return_from_classification(
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
    # If average return is not available, set average return to 0.
    df_avg = _impute_missing_average(
        df=df_avg, col_class=col_class, list_class=list_class,
        month_return=month_return, time=time)
    # loop over each date
    for date in sorted(df_avg[time].unique())[1:]:
        # get last month
        prev_month = np.datetime64(date, 'M') - 1
        # retrieve current and previous month dataframe
        df_curr = df_avg.loc[df_avg[time]==date]
        df_prev = df_avg.loc[df_avg[time]==prev_month]
        # calculate cumulative asset and return
        df_avg.loc[df_avg[time]==date, "cumulative_asset"] = \
            df_prev["cumulative_asset"] * (1 + df_curr[month_return])
        df_avg.loc[df_avg[time]==date, "cumulative_return"] = \
            df_prev["cumulative_asset"] * (1 + df_curr[month_return]) - 1
    return df_avg.reset_index()


def diff_cumulative_return_q5q1(df, var, var_quintile, time="eom"):
    ''' calculate difference in cumulative return between fifth and first quintile (Q5 - Q1)
    Args:
        df: Output of cumulative_return function. Pandas dataframe
        var: variable of interest (ex. AG)
        var_quintile: column name representing quintile of the variable of interest (ex. AG_quintile)
        time: name of column representing time
    Return:
        df_join: dataframe containing the difference in cumulative return between top and bottom quintile
    '''
    # filter by quintile
    df_q1 = df.loc[df[var_quintile]==var+" high"]
    df_q5 = df.loc[df[var_quintile]==var+" low"]
    # sort by time
    df_q1 = df_q1.sort_values(time).set_index(time).add_prefix('q1_')
    df_q5 = df_q5.sort_values(time).set_index(time).add_prefix('q5_')
    # join two dataframes
    df_join = pd.concat([df_q1, df_q5], axis=1, join='inner')
    df_join["q5q1"] = df_join["q5_cumulative_return"] - df_join["q1_cumulative_return"] 
    return df_join


#def diff_cumulative_return_q5q1_groupby(df, var, var_quintile, groupby):
#    '''calculate cumulative return and the difference between first and last quintile for each industry sector
#    Args:
#        df: input Pandas dataframe
#        var: variable of interest (ex. AG)
#        var_quintile: column name representing quintile of the variable of interest (ex. AG_quintile)
#        groupby: column representing industry sector
#    Return:
#        dataframe containing the difference in cumulative return (q1-q5) by industry sector
#    '''
#    df_cum_return_group = {}
#    df_diff_q5q1_group = {}
#    for name, df_group in df.groupby(groupby):
#        io.message("Processing group: %s" %name)
#        io.message(name)
#        # calculate cumulative return
#        df_cum_return_group[name]= cumulative_return_classification(
#            df=df_group, var=var, var_quintile=var_quintile,
#            total_return=total_return, time= date_column)
#        # calculate difference between AG quintile 1 and 5
#        df_diff_q5q1_group[name] = diff_cumulative_return_q5q1(df=df_cum_return_group[name], var=var, var_quintile=var_quintile)
#        
#    for name, df_group in df_diff_q5q1_group.items():
#        # add prefix
#        df_diff_q5q1_group[name] = df_group.add_prefix(str(name)+"_")
#    
#    # concatenate "q5q1" columns from dataframes by industry group
#    return pd.concat([df_group[str(name)+"_q5q1"] for name, df_group in df_diff_q5q1_group.items()], axis=1, join='outer')



    

def calculate_diff_return(df_cum_return, return_label, output_col, time="eom"):
    ''' Calculate difference in return between top and bottom classes.
        Example. Q1 - Q3 or D1 - D10 where 1 is high return.
    Args:
        df_cum_return: cumulative returns calculated by "predict_and_calculate_cum_return".
        return_label: List of return labels in order of [high, medium, low].
        output_col: name of column representing the difference in return
        time: name of time column
    Return:
        df_diff: dataframe containing month, difference in return and "pred" label for plotting
    '''
    # Set time as index
    df_cum_return = df_cum_return.set_index(time)
    # Calculate difference
    df_diff = pd.DataFrame(
        df_cum_return.loc[
            df_cum_return["pred"]==return_label[0]]["cumulative_return"]\
        - df_cum_return.loc[
            df_cum_return["pred"]==return_label[-1]]["cumulative_return"])
    # Assign "Q1+Q2-Q3" as pred value for plotting
    df_diff["pred"] = output_col
    return df_diff


def calculate_cum_return(pred_train, pred_test, list_class, label_fm, time):
    ''' Wrapper of cumulative_return_from_classification function.
    Calculate cumulative return using the prediction.
    Args:
        pred_train, pred_test: prediction of train and test dataset. This is
            an output of utils.prediction function.
        list_class: list of classes. Ex. [0, 1, 2]
        label_fm: monthly return used for calculating cumulative return
        time: column name representing time
    Return:
        df_cum_return_train: cumulative return calculated from train dataset
        df_cum_return_test: cumulative return calculated from test dataset
        model: trained model
    '''
    # Calculate cumulative return
    io.title("Calculating cumulative return")
    io.message(" > Calculating cumulative return of train dataset..")
    df_cum_return_train = cumulative_return_from_classification(
        df=pred_train, list_class=list_class,
        col_class="pred",
        month_return=label_fm, time=time)
    io.message(" > Calculating cumulative return of train dataset..")
    df_cum_return_test = cumulative_return_from_classification(
        df=pred_test, 
        col_class="pred", list_class=list_class,
        month_return=label_fm, time=time)
    return df_cum_return_train, df_cum_return_test

