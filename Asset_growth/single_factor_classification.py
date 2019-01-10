#!/usr/bin/env python
# import common python libraries
from datetime import datetime
import dateutil.relativedelta
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

# set True for development
debug = True

# set features and label
var_interest = 'AG'
features = ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'GS', 'SEV', 'CVROIC', 'FCFA']
categories = ['GICSSubIndustryNumber']    
total_return = "fqTotalReturn"
label = "fqTotalReturn_quintile"
label_map = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}



# select algorithm to run    
perform_eda             = True

# set plot style
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

def evaluate_model(model, df_train, df_test, param_grid={}, n_folds=5):
    ''' evaluate the given model using averaged f1 score after performing grid search.
    Args:
        model: sklearn model
        df_train: train dataset in Pandas dataframe
        df_test: test dataset in Pandas dataframe
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

def learning_curve(model, param_grid, df, train_length, train_end, test_begin, test_end, date_column="eom", file_surfix=""):
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
        f1_train, f1_test = evaluate_model(model, df_train, df_test, param_grid={}, n_folds=5)
        # append to result
        df_result = df_result.append({'h': h, 'f1_train': f1_train, 'f1_test': f1_test}, ignore_index=True)

    # plot learning curve
    plot_learning_curve(df_result, xcol="h", ycols=["f1_train", "f1_test"], title=file_surfix, figsize=(8,8))
    return

def sort_by_var(df, var_interest, month):
    ''' sort samples by variable of interest within each month. A column called var_interest_quintile is added to the dataframe.
        This column represents quiltile of the variable of interest.
    Args:
        df: Pandas dataframe
        var_interest: variable of interest. ex. "AG"
        month: column representing time
    Return:
        df: Pandas dataframe with "var_interest_quintile" column
        dictionary: dictionary with two keys.
            "quintile": name of new column
            "label": map between quintile integer and string name. ex. 0 -> AG1, 1 -> AG2, etc.
    '''
    var_interest_q = var_interest+"_quintile"
    var_label = {x:var_interest+str(x+1) for x in range(0,5)}
    df[var_interest_q] = df.groupby([month])[var_interest]\
                           .transform(lambda x: pd.qcut(x, 5, labels=[4,3,2,1,0]))
    return df, {"quintile":var_interest_q, "label":var_label}

def calculate_cumulative_return(df, single_factor_q, total_return, time="eom"):
    ''' Calculate cumulative return for each quintile (ex. AG1-AG5)
    Args:
        df: Pandas dataframe
        single_factor_q: quintile with respect to the variable of interest (ex. AG1, AG2, etc..)
        total_return: return column
        time: time column
    Return:
        df_avg: Pandas dataframe representing cumulative return for each unit time
    '''
    # calculate average return of each quintile of the variable of interest (ex.AG)
    df_avg = df.groupby([time,single_factor_q])[total_return].mean().reset_index()
    # find the starting month of cumulative return
    first_month = sorted(df_avg[time].unique())[0]
    cumulative_begin_month = np.datetime64(first_month, 'M') - 1
    # add zero return as the beginnig of cumulative return
    for i in range(0,5):
        df_avg = df_avg.append({time:cumulative_begin_month, single_factor_q:float(i), total_return:0.0}, ignore_index=True)
    # create cumulative return column
    df_avg["cumulative_return"] = 0.0
    # loop over each date
    for date in sorted(df_avg[time].unique())[1:]:
        # data from current and previous month
        prev_month = np.datetime64(date, 'M') - 1
        df_prev = df_avg.loc[df_avg[time]==prev_month].sort_values(single_factor_q)
        df_curr = df_avg.loc[df_avg[time]==date].sort_values(single_factor_q)
        # calculate cumulative return
        prev_asset = 1 + df_prev.reset_index()["cumulative_return"]
        curr_return = 1 + df_curr.reset_index()[total_return]
        df_avg.loc[df_avg[time]==date, "cumulative_return"] = np.array(prev_asset * curr_return - 1).tolist()
    return df_avg


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
    for i, ycol in enumerate(ycols):
        df.plot.line(x=xcol, y=ycol, ax=ax, legend=True, marker=i+6, markersize=15)
    # customize plot and save
    ax.set_ylabel("Average F1 score")
    ax.set_xlabel("Training length (months)")
    ax.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig('plots/learning_curve_%s.png' % title)
    return

def plot_groupby_dist(df, x, group_var, hue, hue_str, norm=False, n_bins=50, figsize=(8,8), filename=""):
    ''' plot distribution of given variable for each group. Seperate plot will be generated for each group. 
    Args:
        df: Pandas dataframe
        x: variable to plot
        group_var: categorical variable for group
        hue: additional category. seperate distribution will be plotted for each hue within the same group plot.
        hue_str: dictionary to map hue value and name. i.e. 0 -> Q1, 1 -> Q2, etc.
        norm: normalize distributions
        others: plotting options
    Return: None
    '''
    n_groups = df[group_var].nunique()
    # create figure and axes
    fig, ax = plt.subplots(n_groups, 1, figsize=figsize)
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
        ax[i].set_title(group_name)
        ax[i].grid(False)
        ax[i].legend()
    # customize and save plot
    plt.tight_layout()
    plt.savefig('plots/dist_%s.png' % filename)
    plt.cla()

if False:
    # create train, test dataset
    df_train, df_test = train_test_split(df=df, date_column="eom", 
                                         train_length = 48, 
                                         train_end = to_datetime("2014-12-31"),
                                         test_begin = to_datetime("2015-01-01"),
                                         test_end = to_datetime("2017-10-31"))

    # initiate logistic regression model
    model = LogisticRegression(penalty='l2', random_state=0, multi_class='multinomial', solver='newton-cg', n_jobs=-1)
    param_grid = {'C':np.logspace(0, 4, 5)}

    # evaluate model
    f1_train, f1_test = evaluate_model(model, df_train, df_test, param_grid={}, n_folds=5)



if __name__ == "__main__":

    #------------------------------------------
    # read dataset
    #------------------------------------------
    # read input csv
    df = pd.read_csv(input_path, index_col=None, parse_dates=["eom"])

    # one-hot-encode categorical feature
    #df = pd.get_dummies(df, columns=categories, drop_first=False)

    # sort samples by the variable of interest (AG) within each month and assign quintile
    df, var_int =  sort_by_var(df, var_interest=var_interest, month='eom')

    # make return distribution of each AG group
    ''' make two sets of distributions, normalized and raw'''
    for i, norm in enumerate([True, False]):
        plot_groupby_dist(df=df, x=total_return,\
                                 group_var=categories[0],\
                                 hue=var_int["quintile"], hue_str=var_int["label"], norm=norm,\
                                 n_bins=50, figsize=(8,40), filename="return_sort_by_AG_%s" %i)


    # calculate cumulative return
    calculate_cumulative_return(df=df, single_factor_q=var_interest+"_quintile", total_return=total_return)

    # plot cumulative return
    ''' Add cumulative return plot here. Work in progress '''


    print("Successfully completed all tasks")
