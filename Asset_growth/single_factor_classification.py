#!/usr/bin/env python
# import common python libraries
from datetime import datetime
import dateutil.relativedelta
import itertools
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

#----------------------------------------------
# set user options
#----------------------------------------------
# set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.pd.r3.csv'

# set algorithms to run
run_simple_sort = False
run_classification = True

# set True for development
debug = True

# set features and label
features = ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'GS', 'SEV', 'CVROIC', 'FCFA']
categories = ['GICSSubIndustryNumber']    
total_return = "fqTotalReturn"
#total_return = "fmTotalReturn"
label = "fqTotalReturn_quintile"
label_map = {0.0:'Q1', 1.0:'Q2', 2.0:'Q3', 3.0:'Q4', 4.0:'Q5'}
var = 'AG'

# map industry sector code to string
industry_map={10:"Energy", 15:"Materials", 20:"Industrials", 25:"Consumer discretionary",
              30:"Consumer staples", 35:"Health care", 40:"Financials", 45:"Information technology",
              50:"Communication services", 55:"Utilities", 60:"Real estate", 99:"Unknown"}

# set plot style
markers=('x', 'p', "|", '*', '^', 'v', '<', '>')
lines=("-","--","-.",":")
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
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

def evaluate_model(model, df_train, df_test, features, label, param_grid={}, n_folds=5):
    ''' evaluate the given model using averaged f1 score after performing grid search.
    Args:
        model: sklearn model
        df_train: train dataset in Pandas dataframe
        df_test: test dataset in Pandas dataframe
        features: feature column names
        label: target column name
        param_grid: parameter grid to search
        n_fold: number of cv folds
    Returns:
        f1_train: averaged f1 score from train sample
        f1_test: averaged f1 score from test sample
    '''
    # custom scorer for multi-class classification
    scorer = make_scorer(f1_score, average='macro')
    # run grid search
    cv = GridSearchCV(model, param_grid, cv=n_folds, scoring=scorer, refit=True)
    cv.fit(df_train[features], df_train[label])
    # train using best model
    pred_train = cv.predict(df_train[features])
    pred_test = cv.predict(df_test[features])
    # calculate averaged f1 score
    f1_train = f1_score(df_train[label], pred_train, average='macro')
    f1_test = f1_score(df_test[label], pred_test, average='macro')
    return f1_train, f1_test

def learning_curve(model, param_grid, df, features, label, train_length, train_end, test_begin, test_end, date_column="eom", file_surfix=""):
    ''' given model and dataset, produce learning curves (f1 score vs training length).
    Args:
        model: sklearn model
        param_grid: parameter grid to search
        df: raw input dataframe
        train_length: list of train lengths given in month
        train_end, test_begin, test_end: begin and end date of train, test dataset in datetime format
        date_column: name of column representing time
        file_surfix: file surfix used when saving result
    Return: None
    '''
    print("Creating learning curve for: %s" % file_surfix)
    # dataframe to hold result
    df_result = pd.DataFrame()
    for h in train_length:
        # create train, test dataset
        df_train, df_test = train_test_split(df=df, date_column=date_column,
                                             train_length = h, train_end = train_end,
                                             test_begin = test_begin, test_end = test_end)
        # evaluate model
        f1_train, f1_test = evaluate_model(model, df_train, df_test, features, label, param_grid={}, n_folds=5)
        # append to result
        df_result = df_result.append({'h': h, 'f1_train': f1_train, 'f1_test': f1_test}, ignore_index=True)

    # plot learning curve
    plot_learning_curve(df_result, xcol="h", ycols=["f1_train", "f1_test"], title=file_surfix, figsize=(8,8))
    return

def sort_by_var(df, var, month="eom"):
    ''' sort samples by variable of interest within each month. A column called var_quintile is added to the dataframe.
        This column represents quiltile of the variable of interest.
    Args:
        df: Pandas dataframe
        var: variable of interest. ex. "AG"
        month: column representing time
    Return:
        df: Pandas dataframe with "var_quintile" column
        dictionary: dictionary with two keys.
            "quintile": name of new column
            "label": map between quintile integer and string name. ex. 0 -> AG1, 1 -> AG2, etc.
    '''
    var_quintile = var+"_quintile"
    var_label = {float(x):var+str(x+1) for x in range(0,5)}
    df[var_quintile] = df.groupby([month])[var]\
                           .transform(lambda x: pd.qcut(x, 5, labels=[4,3,2,1,0]))
    return df, var_quintile, var_label

def cumulative_return(df, var_quintile, total_return, time="eom"):
    ''' Calculate cumulative return of each quintile (ex. AG1-AG5)
    Args:
        df: Pandas dataframe
        var_quintile: quintile with respect to the variable of interest (ex. AG1, AG2, etc..)
        total_return: return column
        time: time column
    Return:
        df_avg: Pandas dataframe representing cumulative return for each unit time
    '''
    def _impute_missing_average(df, var_quintile, total_return, time):
        ''' check if average return available. If not, set average return to 0. '''
        for date in sorted(df[time].unique())[1:]:
            df_curr = df.loc[df[time]==date].sort_values(var_quintile) # retrieve current return and previous asset
            for i in range(0,5): 
                if i not in df_curr[var_quintile].tolist():
                    print("Found a period in which mean return is not available: date=%s, quintile=%s" %(date, i))
                    df_add = df_curr.head(1).copy()
                    df_add[total_return]=0
                    df_add[var_quintile]=i
                    df = pd.concat([df, df_add], axis=0, ignore_index=True).sort_values(var_quintile)
        return df
    # calculate average return of each quintile of the variable of interest (ex.AG)
    df_avg = df.groupby([time,var_quintile])[total_return].mean().reset_index()
    # find the starting month of cumulative return
    first_month = sorted(df_avg[time].unique())[0]
    cumulative_begin_month = np.datetime64(first_month, 'M') - 1
    # add zero return as the beginnig of cumulative return
    for i in range(0,5):
        df_avg = df_avg.append({time:cumulative_begin_month, var_quintile:float(i), total_return:0.0}, ignore_index=True)
    # create cumulative return column
    df_avg["cumulative_asset"] = 1.0
    df_avg["cumulative_return"] = 0.0
    # if average return is not available, set average return to 0.
    df_avg = _impute_missing_average(df=df_avg, var_quintile=var_quintile, total_return=total_return, time=time)
    # loop over each date
    for date in sorted(df_avg[time].unique())[1:]:
        # data from current and previous month
        prev_month = np.datetime64(date, 'M') - 1
        df_prev = df_avg.loc[df_avg[time]==prev_month].sort_values(var_quintile)
        df_curr = df_avg.loc[df_avg[time]==date].sort_values(var_quintile) # retrieve current return and previous asset
        curr_return = df_curr.reset_index()[total_return]
        prev_asset = df_prev.reset_index()["cumulative_asset"]
        # calculate cumulative asset
        df_avg.loc[df_avg[time]==date, "cumulative_asset"] = np.array(prev_asset * (1 + curr_return)).tolist()
        df_avg.loc[df_avg[time]==date, "cumulative_return"] = np.array(prev_asset * (1 + curr_return) - 1).tolist()
    return df_avg

def diff_cumulative_return_q5q1(df, var_quintile, time="eom"):
    ''' calculate difference in cumulative return between fifth and first quintile (Q5 - Q1)
    Args:
        df: Output of cumulative_return function. Pandas dataframe
        time: name of column representing time
    Return:
        df_join: dataframe containing the difference in cumulative return between top and bottom quintile
    '''
    # filter by quintile
    df_q1 = df.loc[df[var_quintile]==0.0]
    df_q5 = df.loc[df[var_quintile]==4.0]
    # sort by time
    df_q1 = df_q1.sort_values(time).set_index(time).add_prefix('q1_')
    df_q5 = df_q5.sort_values(time).set_index(time).add_prefix('q5_')
    # join two dataframes
    df_join = pd.concat([df_q1, df_q5], axis=1, join='inner')
    df_join["q5q1"] = df_join["q5_cumulative_return"] - df_join["q1_cumulative_return"] 
    return df_join


def diff_cumulative_return_q5q1_groupby(df, var_quintile):
    '''calculate cumulative return and the difference between first and last quintile for each industry sector
    Args: input Pandas dataframe
    Return: dataframe containing the difference in cumulative return (q1-q5) by industry sector
    '''
    df_cum_return_group = {}
    df_diff_q5q1_group = {}
    for name, df_group in df.groupby(categories[0]):
        print("Processing group: %s" %name)
        # calculate cumulative return
        df_cum_return_group[name]= cumulative_return(df=df_group, var_quintile=var_quintile, total_return=total_return)
        # calculate difference between AG quintile 1 and 5
        df_diff_q5q1_group[name] = diff_cumulative_return_q5q1(df=df_cum_return_group[name], var_quintile=var_quintile)
        
    for name, df_group in df_diff_q5q1_group.items():
        # add prefix
        df_diff_q5q1_group[name] = df_group.add_prefix(str(name)+"_")
    
    # concatenate "q5q1" columns from dataframes by industry group
    return pd.concat([df_group[str(name)+"_q5q1"] for name, df_group in df_diff_q5q1_group.items()], axis=1, join='outer')


#----------------------------------------------
# plotting functions
#----------------------------------------------

def plot_learning_curve(df, xcol="h", ycols=["f1_train", "f1_test"], title="", figsize=(8,8)):
    ''' plot learning curve and save as png
    Args:
        df: dataframe containing model score and training length
        xcol: name of column representing training length
        ycols: list of column names representing model scores
        others: plotting options
    Return: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # make plot
    line = itertools.cycle(lines) 
    for i, ycol in enumerate(ycols):
        df.plot.line(x=xcol, y=ycol, ax=ax, legend=True, linestyle=next(line))
    # customize plot and save
    ax.set_ylabel("Average F1 score")
    ax.set_xlabel("Training length (months)")
    ax.set_ylim(0,0.4)
    plt.tight_layout()
    plt.savefig('plots/learning_curve_%s.png' % title)
    return

def plot_dist_groupby_hue(df, x, group_var, group_title, hue, hue_str, norm=False, n_subplot_columns=1, n_bins=50, figsize=(20,16), filename=""):
    ''' plot distribution of given variable for each group. Seperate plot will be generated for each group. 
    Args:
        df: Pandas dataframe
        x: variable to plot
        group_var: categorical variable for group
        group_title: dictionary that maps group variable to human-recognizable title (45 -> Information technology)
        hue: additional category. seperate distribution will be plotted for each hue within the same group plot.
        hue_str: dictionary to map hue value and name. i.e. 0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Return: None
    '''
    n_groups = df[group_var].nunique()
    # create figure and axes
    n_subplot_rows = round(n_groups / n_subplot_columns)
    fig, ax = plt.subplots(n_subplot_rows, n_subplot_columns, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    for i, group_name in enumerate(sorted(df[group_var].unique())):
        # filter group
        df_group = df.loc[df[group_var] == group_name]
        n_hue = df[hue].nunique()
        # loop over hue
        for j, hue_name in enumerate(sorted(df_group[hue].unique())):
            df_hue = df_group.loc[df_group[hue] == hue_name]
            df_hue[x].hist(bins=n_bins, alpha=0.6, ax=ax[i], range=(-1,1), edgecolor="black", label=hue_str[hue_name], density=norm)
        # customize plot
        ax[i].set_xlabel(x)
        ax[i].set_ylabel("n")
        ax[i].set_title(group_title[i])
        ax[i].grid(False)
        ax[i].legend()
    # customize and save plot
    ax = ax.reshape(n_subplot_rows, n_subplot_columns)
    plt.tight_layout()
    plt.savefig('plots/dist_%s.png' % filename)
    plt.cla()
    return

def plot_dist_hue(df, x, hue, hue_str, norm=False, n_bins=50, figsize=(8,5), filename="", alpha=0.6):
    ''' plot distribution of given variable for each group.
    Args:
        df: Pandas dataframe
        x: variable to plot
        hue: additional category. seperate distribution will be plotted for each hue within the same plot.
        hue_str: dictionary to map hue value and name. i.e. 0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Return: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # filter group
    n_hue = df[hue].nunique()
    # loop over hue
    for j, hue_name in enumerate(sorted(df[hue].unique())):
        df_hue = df.loc[df[hue] == hue_name]
        #df_hue[x].hist(bins=n_bins, alpha=alpha, ax=ax, range=(-1,1), edgecolor="black", label=hue_str[hue_name], density=norm)
        df_hue[x].hist(bins=n_bins, histtype='step', ax=ax, range=(-1,1), label=hue_str[hue_name], density=norm, linewidth=1.5)
    # customize plot
    ax.set_xlabel(x)
    ax.set_ylabel("n")
    ax.grid(False)
    ax.legend()
    # customize and save plot
    plt.tight_layout()
    plt.savefig('plots/dist_%s.png' % filename)
    plt.cla()

def plot_line_groupby(df, x, y, groupby, group_label, ylog=False, x_label="", y_label="", figsize=(20,6), filename=""):
    ''' create line plot for different group in the same axes.
    Args:
        df: Pandas dataframe
        x: name of column used for x
        y: name of column to plot
        groupby: column representing different groups
        group_label: dictionary that maps gruop value to title. ex. {0:"AG1", 1:"AG2", etc.}
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax=ax.flatten()
    line = itertools.cycle(lines) 
    for name, df_group in df.groupby(groupby):
        if x=="index":
            df_group[y].plot(kind='line', legend=True, label=group_label[name], linewidth=2.0, linestyle=next(line))
        else:
            df_group.set_index(x)[y].plot(kind='line', legend=True, label=group_label[name], linewidth=2.0, linestyle=next(line))
    # customize and save plot
    if ylog:
        ax[0].set_yscale('log')
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    #ax[0].grid(False)
    plt.tight_layout()
    plt.savefig('plots/line_%s.png' % filename)
    plt.cla()

def plot_line_multiple_cols(df, x, list_y, legends, x_label, y_label, figsize=(20,6), filename=""):
    ''' create line plot from multiple columns in the same axes.
    Args:
        df: Pandas dataframe
        x: name of column used for x
        list_y: list of column names to plot
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax=ax.flatten()
    line = itertools.cycle(lines) 
    for i, y in enumerate(list_y):
        if x=="index":
            df[y].plot(kind='line', linewidth=2.0, label=legends[i], linestyle=next(line))
        else:
            df.set_index(x)[y].plot(kind='line', linewidth=2.0, label=legends[i], linestyle=next(line))
    # customize and save plot
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(x_label)
    #ax[0].grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/line_%s.png' % filename)
    plt.cla()
    return
    
def plot_heatmap(df, x_label, y_label, figsize=(20,6), filename=""):
    ''' create heatmap from given dataframe
    Args:
        df: Pandas dataframe
        others: plotting options
    Return:
        None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # plot heatmap
    ax = sns.heatmap(df, annot=True, cmap="RdBu_r")
    # customize and save plot
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig('plots/heatmap_%s.png' % filename)
    plt.cla()
    return

if __name__ == "__main__":

    #------------------------------------------
    # read dataset
    #------------------------------------------
    # read input csv
    df = pd.read_csv(input_path, index_col=None, parse_dates=["eom"])

    #------------------------------------------
    # classification by simple sort
    #------------------------------------------
    ''' classify samples by single factor quintile (ex. AG1-AG5). Cumulative returns are calculated.
        Also, the difference in the cumulative returns between fist and last quintiles are calculated.
    '''
    if run_simple_sort:

        # sort samples by the variable of interest (AG) within each month and assign quintile
        df, var_quintile, var_label =  sort_by_var(df, var=var)

        # calculate average return by industry sector
        df_return_mean = df.groupby(['AG_quintile', 'GICSSubIndustryNumber']).mean()['fmTotalReturn']\
                           .unstack(1).transpose().rename(industry_map, axis=0)\
                           .rename({0:"AG1", 1:"AG2", 2:"AG3", 3:"AG4", 4:"AG5"}, axis=1)
        df_return_std = df.groupby(['AG_quintile', 'GICSSubIndustryNumber']).std()['fmTotalReturn']\
                          .unstack(1).transpose().rename(industry_map, axis=0)\
                          .rename({0:"AG1", 1:"AG2", 2:"AG3", 3:"AG4", 4:"AG5"}, axis=1)

        # calculate cumulative return
        df_cum_return = cumulative_return(df=df, var_quintile=var_quintile, total_return=total_return)

        # calculate difference between AG quintile 1 and 5
        df_diff_q5q1 = diff_cumulative_return_q5q1(df=df_cum_return, var_quintile=var_quintile)

        # calculate difference between AG quintile 1 and 5 by industry sector
        df_diff_q5q1_groupby = diff_cumulative_return_q5q1_groupby(df=df, var_quintile=var_quintile)

        #------------------------------------------
        # make plots for simple sort method
        #------------------------------------------
        ''' make plots for simple sort method.
        '''
        # make AG distribution of each AG quintile
        plot_dist_hue(df=df, x=var, hue=var_quintile, hue_str=var_label, norm=False, \
                      filename="AG_%s" %(total_return))

        # make return distribution of all industry sector combined
        plot_dist_hue(df=df, x=total_return, hue=var_quintile, hue_str=var_label, norm=True, \
                      filename="return_all_sectors_%s" %(total_return), alpha=0.4)

        # make return distribution by industry sector
        for i, norm in enumerate([True, False]):
            plot_dist_groupby_hue(df=df, x=total_return, group_var=categories[0],
                                  group_title=[industry_map[int(x[:-5])] for x in df_diff_q5q1_groupby.columns],\
                                  hue=var_quintile, hue_str=var_label, norm=norm, n_subplot_columns=4,\
                                  filename="return_sort_by_AG_%s_%s" %(total_return, i))

        # plot heatmap of mean and standard deviation of return
        plot_heatmap(df=df_return_mean, x_label="AG quintile", y_label="Industry", figsize=(10,7), filename="mean_%s" % total_return)
        plot_heatmap(df=df_return_std, x_label="AG quintile", y_label="Industry", figsize=(10,7), filename="std_%s" % total_return)


        # plot cumulative return by AG quintile
        plot_line_groupby(df=df_cum_return,\
                          x="eom", y="cumulative_return",\
                          groupby=var_quintile, group_label = var_label,\
                          x_label="Time", y_label="Cumulative %s" %total_return, figsize=(15,6), filename = "cum_%s_sort" % total_return)

        # plot monthly return by AG quintile. only plot AG1 and AG5
        plot_line_groupby(df=df_cum_return.loc[(df_cum_return[var_quintile]==0.0) | (df_cum_return[var_quintile]==4.0)],
                          x="eom", y=total_return,
                          groupby=var_quintile, group_label = {0.0:'AG1', 4.0:'AG5'},\
                          x_label="Time", y_label="Monthly %s" %total_return, figsize=(15,6), filename = "month_%s_sort" % total_return)

        # plot difference between AG1 and AG5 in cumulative return
        plot_line_multiple_cols(df=df_diff_q5q1, x="index", list_y=["q5q1"], legends=["All industry"], x_label="Time", \
                           y_label="Cumulative %s (Q5-Q1)" %total_return, figsize=(20,6), filename="diff_cum_q5q1_%s" % total_return)

        # plot difference between AG1 and AG5 in cumulative return by industry
        plot_line_multiple_cols(df=df_diff_q5q1_groupby, x="index", list_y=df_diff_q5q1_groupby.columns,\
                           legends=[industry_map[int(x[:-5])] for x in df_diff_q5q1_groupby.columns], \
                           x_label="Time", y_label="Cumulative %s (Q5-Q1)" %total_return, figsize=(20,6), filename="diff_cum_q5q1_industry_%s" % total_return)


    #------------------------------------------
    # single factor portfolio by classification
    #------------------------------------------
    ''' classify samples using AG and industry as features and return quintile (Q1-Q5) as labels.
    '''
    if run_classification:

        # one-hot-encode categorical feature
        df_ml = pd.get_dummies(data=df, columns=[categories[0]], drop_first=False)

        # create train, test dataset
        df_train, df_test = train_test_split(df=df_ml, date_column="eom", 
                                             train_length = 48, 
                                             train_end = to_datetime("2014-12-31"),
                                             test_begin = to_datetime("2015-01-01"),
                                             test_end = to_datetime("2017-10-31"))

        # initiate logistic regression model
        model = LogisticRegression(penalty='l2', random_state=0, multi_class='multinomial', solver='newton-cg', n_jobs=-1)
        param_grid = {'C':np.logspace(0, 1, 5)}

        # evaluate model
        f1_train, f1_test = evaluate_model(model, df_train, df_test,
                                           features=['AG'] + ["GICSSubIndustryNumber_"+str(x) for x in industry_map],
                                           label=label, param_grid={}, n_folds=5)


        # make learning curve
        learning_curve(model, param_grid, df_ml,
                       features=['AG'] + ["GICSSubIndustryNumber_"+str(x) for x in industry_map], label=label,
                       train_length=[6,9,12,24,48,96],
                       train_end=to_datetime("2014-12-31"),
                       test_begin=to_datetime("2015-01-01"),
                       test_end=to_datetime("2017-10-31"), date_column="eom", file_surfix="linear")


        # make AG distribution of each return quintile (Q1-Q5)
        plot_dist_hue(df=df, x=var, hue=label, hue_str=label_map, norm=False, \
                      filename="AG_by_%s_quintile" %(total_return))





    print("Successfully completed all tasks")
