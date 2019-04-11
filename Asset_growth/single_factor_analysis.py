#!/usr/bin/env python
# import common python libraries
from datetime import datetime
import dateutil.relativedelta
import itertools
import numpy as np
import pandas as pd
from _plots import *
from _backtest import *
from _ML import *

#----------------------------------------------
# set user options
#----------------------------------------------
# set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.pd.r4.csv'

# set algorithms to run
run_simple_sort = False
run_ag_fc = True
run_classification = False

# set True for development
debug = True

# set features
features = ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'GS', 'SEV', 'CVROIC', 'FCFA']
categories = ['GICSSubIndustryNumber']    
# return label
total_return = "fqTotalReturn" # "fmTotalReturn"
month_return = "fmTotalReturn"
return_quintile = total_return+"_quintile"
return_tertile = total_return+"_tertile"
return_quintile_map = {0.0:'Q1', 1.0:'Q2', 2.0:'Q3', 3.0:'Q4', 4.0:'Q5'}

# set variable of interest
var = 'AG'
var_quintile = var+"_quintile"
var_tertile = var+"_tertile"
var_label = {float(x):var+str(x+1) for x in range(0,5)} # ex. {0: AG1, 1:AG2, etc.}

# map industry sector code to string
sector_map={10:"Energy", 15:"Materials", 20:"Industrials", 25:"Consumer discretionary",
              30:"Consumer staples", 35:"Health care", 40:"Financials", 45:"Information technology",
              50:"Communication services", 55:"Utilities", 60:"Real estate", 99:"Unknown"}

#----------------------------------------------
# define functions
#----------------------------------------------

def to_datetime(date):
    ''' convert given string to datetime"
    Args:
        date: date given in "YYYY-MM-DD" format
    Return:
        date in datetime format
    '''
    return datetime.strptime(date, '%Y-%m-%d')

def train_test_split(df, date_column, train_length, train_end, test_begin, test_end):
    ''' create train and test dataset.
    Args:
        df: pandas dataframe
        date_column: date column in datetime format
        train_length: length of train length in month
        train_end: train end month in datetime format
        test_begin: test begin month in datetime format
        test_end: test end month in datetime format
    Return:
        df_train: train dataset
        df_test: test dataset
    '''
    # find train begin date using train length
    train_begin = train_end - dateutil.relativedelta.relativedelta(months=train_length)
    # create train and test dataset 
    df_train = df.loc[(df[date_column] >= train_begin) & (df[date_column] <= train_end)]
    df_test = df.loc[(df[date_column] >= test_begin) & (df[date_column] <= test_end)]
    return df_train, df_test

#def evaluate_model(model, df_train, df_test, features, label, param_grid={}, n_folds=5):
#    ''' evaluate the given model using averaged f1 score after performing grid search.
#    Args:
#        model: sklearn model
#        df_train: train dataset in Pandas dataframe
#        df_test: test dataset in Pandas dataframe
#        features: feature column names
#        label: target column name
#        param_grid: parameter grid to search
#        n_fold: number of cv folds
#    Returns:
#        f1_train: averaged f1 score from train sample
#        f1_test: averaged f1 score from test sample
#    '''
#    # custom scorer for multi-class classification
#    scorer = make_scorer(f1_score, average='macro')
#    # run grid search
#    cv = GridSearchCV(model, param_grid, cv=n_folds, scoring=scorer, refit=True)
#    cv.fit(df_train[features], df_train[label])
#    # train using best model
#    pred_train = cv.predict(df_train[features])
#    pred_test = cv.predict(df_test[features])
#    # calculate averaged f1 score
#    f1_train = f1_score(df_train[label], pred_train, average='macro')
#    f1_test = f1_score(df_test[label], pred_test, average='macro')
#    return f1_train, f1_test
#
#def learning_curve(model, param_grid, df, features, label, train_length, train_end, test_begin, test_end, date_column="eom", file_surfix=""):
#    ''' given model and dataset, produce learning curves (f1 score vs training length).
#    Args:
#        model: sklearn model
#        param_grid: parameter grid to search
#        df: raw input dataframe
#        train_length: list of train lengths given in month
#        train_end, test_begin, test_end: begin and end date of train, test dataset in datetime format
#        date_column: name of column representing time
#        file_surfix: file surfix used when saving result
#    Return: None
#    '''
#    print("Creating learning curve for: %s" % file_surfix)
#    # dataframe to hold result
#    df_result = pd.DataFrame()
#    for h in train_length:
#        # create train, test dataset
#        df_train, df_test = train_test_split(df=df, date_column=date_column,
#                                             train_length = h, train_end = train_end,
#                                             test_begin = test_begin, test_end = test_end)
#        # evaluate model
#        f1_train, f1_test = evaluate_model(model, df_train, df_test, features, label, param_grid={}, n_folds=5)
#        # append to result
#        df_result = df_result.append({'h': h, 'f1_train': f1_train, 'f1_test': f1_test}, ignore_index=True)
#
#    # plot learning curve
#    plot_learning_curve(df_result, xcol="h", ycols=["f1_train", "f1_test"], title=file_surfix, figsize=(8,8))
#    return

#def discretize_variables_by_month(df, variables, month="eom"):
#    ''' discretize variables by assigning a quintile and tertile class within each month. 
#    Args:
#        df: Pandas dataframe containing variables
#        variables: list of variables to discretize
#        month: column representing time
#    Return:
#        df: Pandas dataframe with columns named x_quintile, x_tertile for all variable x.
#    '''
#    # create classification labels
#    for var in variables:
#        df[var+"_tertile"] = df.groupby([month])[var].transform(lambda x: pd.qcut(x, 3, labels=[var+" low", var+" mid", var+" high"]))
#        df[var+"_quintile"] = df.groupby([month])[var].transform(lambda x: pd.qcut(x, 5, labels=[var+" low", var+" mid-low", var+" mid", var+" mid-high", var+" high"]))
#    return df

#def cumulative_return(df, var, var_quintile, total_return, time="eom"):
#    ''' Calculate cumulative return of each quintile (ex. AG1-AG5)
#    Args:
#        df: Pandas dataframe
#        var: variable of interest (ex. AG)
#        var_quintile: column name representing quintile of the variable of interest (ex. AG_quintile)
#        total_return: return column
#        time: time column
#    Return:
#        df_avg: Pandas dataframe representing cumulative return for each unit time
#    '''
#    def _impute_missing_average(df, var, var_quintile, total_return, time):
#        ''' check if average return available. If not, set average return to 0. '''
#        for date in sorted(df[time].unique())[1:]:
#            df_curr = df.loc[df[time]==date].sort_values(var_quintile) # retrieve current return and previous asset
#            for quintile in [var+" low", var+" mid-low", var+" mid", var+" mid-high", var+" high"]:
#                #if quintile not in df[var_quintile].unique():
#                if quintile not in df_curr[var_quintile].unique():
#                    print("Found a period in which mean return is not available: date=%s, quintile=%s" %(date, quintile))
#                    df_add = df_curr.head(1).copy()
#                    df_add[total_return]=0
#                    df_add[var_quintile]=quintile
#                    df = pd.concat([df, df_add], axis=0, ignore_index=True).sort_values(var_quintile)
#        return df.set_index(var_quintile)
#    # calculate average return of each quintile of the variable of interest (ex.AG)
#    df_avg = df.groupby([time,var_quintile])[total_return].mean().reset_index()
#    # find the starting month of cumulative return
#    first_month = sorted(df_avg[time].unique())[0]
#    cumulative_begin_month = np.datetime64(first_month, 'M') - 1
#    # add zero return as the beginnig of cumulative return
#    for quintile in [var+" low", var+" mid-low", var+" mid", var+" mid-high", var+" high"]:
#        df_avg = df_avg.append({time:cumulative_begin_month, var_quintile:quintile, total_return:0.0}, ignore_index=True)
#    # create cumulative return column
#    df_avg["cumulative_asset"] = 1.0
#    df_avg["cumulative_return"] = 0.0
#    # if average return is not available, set average return to 0.
#    df_avg = _impute_missing_average(df=df_avg, var=var, var_quintile=var_quintile, total_return=total_return, time=time)
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

def diff_cumulative_return_q5q1(df, var, var_classes, time="eom"):
    ''' calculate difference in cumulative return between fifth and first quintile (Q5 - Q1)
    Args:
        df: Output of cumulative_return function. Pandas dataframe
        var: variable of interest (ex. AG)
        var_classes: column name representing quintile of the variable of interest (ex. AG_quintile)
        time: name of column representing time
    Return:
        df_join: dataframe containing the difference in cumulative return between top and bottom quintile
    '''
    # filter by quintile
    df_q1 = df.loc[df[var_classes]==var+" high"]
    df_q5 = df.loc[df[var_classes]==var+" low"]
    # sort by time
    df_q1 = df_q1.sort_values(time).set_index(time).add_prefix('q1_')
    df_q5 = df_q5.sort_values(time).set_index(time).add_prefix('q5_')
    # join two dataframes
    df_join = pd.concat([df_q1, df_q5], axis=1, join='inner')
    df_join["q5q1"] = df_join["q5_cumulative_return"] - df_join["q1_cumulative_return"] 
    return df_join


def diff_cumulative_return_q5q1_groupby(df, var, var_classes, groupby, total_return):
    '''calculate cumulative return and the difference between first and last quintile for each industry sector
    Args:
        df: input Pandas dataframe
        var: variable of interest (ex. AG)
        var_classes: column name representing quintile of the variable of interest (ex. AG_quintile)
        groupby: column representing industry sector
    Return:
        dataframe containing the difference in cumulative return (q1-q5) by industry sector
    '''
    df_cum_return_group = {}
    df_diff_q5q1_group = {}
    for name, df_group in df.groupby(groupby):
        print("Processing group: %s" %name)
        print(name)
        # calculate cumulative return
        df_cum_return_group[name]= cumulative_return(df=df_group, var_classes=var_classes, total_return=total_return)
        # calculate difference between AG quintile 1 and 5
        df_diff_q5q1_group[name] = diff_cumulative_return_q5q1(df=df_cum_return_group[name], var=var, var_classes=var_classes)
        
    for name, df_group in df_diff_q5q1_group.items():
        # add prefix
        df_diff_q5q1_group[name] = df_group.add_prefix(str(name)+"_")
    
    # concatenate "q5q1" columns from dataframes by industry group
    return pd.concat([df_group[str(name)+"_q5q1"] for name, df_group in df_diff_q5q1_group.items()], axis=1, join='outer')


if __name__ == "__main__":

    #------------------------------------------
    # read dataset
    #------------------------------------------
    # read input csv
    df = pd.read_csv(input_path, index_col=None, parse_dates=["eom"])

    # assign quintile and tertile classes to AG, return, and FCFA
    df = discretize_variables_by_month(df=df, variables=[var, total_return, 'FCFA'], month="eom")

    #------------------------------------------
    # classification by simple sort (AG and industry)
    #------------------------------------------
    ''' classify samples by single factor quintile (ex. AG1-AG5). Cumulative returns are calculated.
        Also, the difference in the cumulative returns between fist and last quintiles are calculated.
    '''
    if run_simple_sort:

        # calculate average return by industry sector
        df_return_mean = df.groupby(['AG_tertile', 'GICSSubIndustryNumber']).mean()[month_return]\
                           .unstack(1).transpose().rename(sector_map, axis=0)
        df_return_std = df.groupby(['AG_tertile', 'GICSSubIndustryNumber']).std()[month_return]\
                          .unstack(1).transpose().rename(sector_map, axis=0)

        # calculate cumulative return
        df_cum_return = cumulative_return(df=df, var_classes=var_tertile, total_return=month_return)

        # calculate difference between AG top and bottom class
        df_diff_q5q1 = diff_cumulative_return_q5q1(df=df_cum_return, var=var, var_classes=var_tertile)

        # calculate difference between AG top and bottom class by industry sector
        df_diff_q5q1_groupby = diff_cumulative_return_q5q1_groupby(df=df, var=var, var_classes=var_tertile, total_return=month_return, groupby=categories[0])

        #------------------------------------------
        # make plots for simple sort method
        #------------------------------------------
        ''' make plots for simple sort method.
        '''
        # make AG distribution of each AG quintile
        plot_dist_hue(df=df, x=var, hue=var_tertile, hue_str={x:x for x in df[var_tertile].unique()}, norm=False, \
                      filename="AG")

        # make return distribution of all industry sector combined
        plot_dist_hue(df=df, x=total_return, hue=var_tertile, hue_str=var_label, norm=True, \
                      filename="return_all_sectors_%s" %(total_return), alpha=0.4)

        # make return distribution by industry sector
        for i, norm in enumerate([True, False]):
            plot_dist_groupby_hue(df=df, x=total_return, group_var=categories[0],
                                  group_title=[sector_map[int(x[:-5])] for x in df_diff_q5q1_groupby.columns],\
                                  hue=var_tertile, hue_str=var_label, norm=norm, n_subplot_columns=4,\
                                  filename="return_sort_by_AG_%s_%s" %(total_return, i))

        # plot cumulative return by AG quintile
        plot_line_groupby(df=df_cum_return,\
                          x="eom", y="cumulative_return",\
                          groupby=var_tertile, group_label = {var:var for var in df_cum_return[var_tertile].unique()},\
                          x_label="Time", y_label="Cumulative %s" %month_return, ylog=False, figsize=(15,6), filename = "cum_%s_sort" % month_return)

        # plot monthly return by AG quintile. only plot AG1 and AG5
        plot_line_groupby(df=df_cum_return.loc[(df_cum_return[var_tertile]==var+" low") | (df_cum_return[var_tertile]==var+" high")],
                          x="eom", y=month_return, groupby=var_tertile, group_label = {var:var for var in df_cum_return[var_tertile].unique()},\
                          x_label="Time", y_label="Monthly %s" %month_return, figsize=(15,6), filename = "month_%s_sort" % month_return)

        # plot difference between AG1 and AG5 in cumulative return
        plot_line_multiple_cols(df=df_diff_q5q1, x="index", list_y=["q5q1"], legends=["All industry"], x_label="Time", ylog=False,\
                           y_label="Cumulative %s\n(%s low - high)" % (month_return, var), figsize=(8,6), filename="diff_cum_q5q1_%s" % month_return)

        # plot difference between AG1 and AG5 in cumulative return by industry
        plot_line_multiple_cols(df=df_diff_q5q1_groupby, x="index", list_y=df_diff_q5q1_groupby.columns,\
                           legends=[sector_map[int(x[:-5])] for x in df_diff_q5q1_groupby.columns], ylog=False,\
                           x_label="Time", y_label="Cumulative %s\n(%s low - high)" % (month_return, var), figsize=(8,6), filename="diff_cum_q5q1_industry_%s" % month_return)

        #------------------------------------------
        # heatmaps
        #------------------------------------------
        # plot heatmap of mean and standard deviation of return
        plot_heatmap(df=df_return_mean, x_label="AG quintile", y_label="Industry", figsize=(10,7), filename="mean_%s" % total_return)
        plot_heatmap(df=df_return_std, x_label="AG quintile", y_label="Industry", figsize=(10,7), filename="std_%s" % total_return)

        # plot heatmap of number of samples within AG and return quintile groups
        plot_heatmap(df=df.groupby([var_quintile, return_quintile]).count().iloc[:,0].unstack(level=-1),\
                     x_label=total_return, y_label="AG",\
                     figsize=(10,7), filename="AG_%s_quintile" % total_return, fmt='.0f')

        # plot heatmap of number of samples within AG and return tertile groups 
        plot_heatmap(df=df.groupby([var_tertile, return_tertile]).count().iloc[:,0].unstack(level=-1),\
                     x_label=total_return, y_label="AG",\
                     figsize=(10,7), filename="AG_%s_tertile" % total_return, fmt='.0f')

    #------------------------------------------
    # AG, FCFA, sector
    #------------------------------------------

    if run_ag_fc == True:

        # plot return by AG and FCFA
        plot_box(df=df, x=var_tertile, y=total_return, title="", color="white", linewidth=1, showmeans=True, ylim=(-0.5,2),
                 hue="FCFA_tertile", #ylim=(0.7, 1.3),
                 x_label="AG", y_label=total_return, figsize=(10,6), filename="%s_by_AG_FCFA" %total_return)


        # plot heatmap of number of samples within FCFA and return tertile groups for given AG group
        for i_ag in df[var_tertile].unique():
            plot_heatmap(df=df.loc[df[var_tertile]==i_ag].groupby([return_tertile, "FCFA_tertile"]).count().iloc[:,0].unstack(level=-1),\
                         x_label="fq total return", y_label="FCFA",\
                         figsize=(10,7), filename="%s_FCFA_AG%s" % (total_return, i_ag), fmt='.0f')

        # plot average return of AG and FCFA tertile group
        plot_heatmap(df=df.groupby([var_tertile, "FCFA_tertile"]).mean()[total_return].unstack(1),\
                         x_label="FCFA", y_label="AG",\
                         figsize=(10,7), filename="%s_FCFA_AG_tertile" % (total_return), fmt='.3f')

        # plot standard deviation of return for AG and FCFA tertile group by sector
        df_std_list = {}
        for i_industry in df[categories[0]].unique():
            df_std_list[i_industry]=df.loc[df[categories[0]]==i_industry]\
                                          .groupby([var_tertile, "FCFA_tertile"])\
                                          .std()[total_return].unstack(1)
        plot_heatmap_group(df_list=df_std_list, n_subplot_columns=4,
                           x_label="FCFA", y_label="AG", group_map=sector_map, figsize=(25,20),
                           filename="%s_std_FCFA_AG_sector_tertile" % (total_return), fmt='.3f', cmap=sns.light_palette("gray"))

        # plot average return of AG and FCFA tertile group by sector
        df_mean_list = {}
        for i_industry in df[categories[0]].unique():
            df_mean_list[i_industry]=df.loc[df[categories[0]]==i_industry]\
                                          .groupby([var_tertile, "FCFA_tertile"])\
                                          .mean()[total_return].unstack(1)
        plot_heatmap_group(df_list=df_mean_list, df_err_list=df_std_list, n_subplot_columns=4,
                           x_label="FCFA", y_label="AG", group_map=sector_map, figsize=(25,20), fmt="s", cmap=sns.color_palette("RdBu_r", 7),
                           filename="%s_mean_FCFA_AG_sector_tertile" % (total_return))




    print("Successfully completed all tasks")
