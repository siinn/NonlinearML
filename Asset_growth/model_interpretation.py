#!/usr/bin/env python
# import common python libraries
from datetime import datetime
import dateutil.relativedelta
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import pandas as pd
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
features = ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'GS', 'SEV', 'CVROIC', 'FCFA']
categories = ['GICSSubIndustryNumber']    
label = "fqTotalReturn_quintile"

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
    df = pd.get_dummies(df, columns=categories, drop_first=False)

    #------------------------------------------
    # logistic regression
    #------------------------------------------
    ''' run logistic regression for multi-class classification.
    There are 5 classes (quintile) created from forward 3 month return.
    Learning curves are plotted.
    '''
    # initiate logistic regression model
    model = LogisticRegression(penalty='l2', random_state=0, multi_class='multinomial', solver='newton-cg', n_jobs=-1)
    param_grid = {'C':np.logspace(0, 4, 5)}

    # create learning curve for linear model
    learning_curve(model=model,
                   param_grid=param_grid,
                   df=df,
                   train_length = [3,6,12,24,48,96,240], 
                   train_end = to_datetime("2014-12-31"),
                   test_begin = to_datetime("2015-01-01"),
                   test_end = to_datetime("2017-10-31"),
                   file_surfix="linear")

    #------------------------------------------
    # xgboost
    #------------------------------------------
    ''' run xgboost for multi-class classification.
    There are 5 classes (quintile) created from forward 3 month return.
    Learning curves are plotted.
    '''
    # initiate xgboost
    model = XGBClassifier(n_jobs=-1, random_state=0)
    param_grid = { 'max_depth': [3],
                   'learning_rate': np.logspace(0, 0.1, 5),
                   'n_estimators': [50],
                   'subsample': [0.5, 1],
                   'objective': 'multi:softmax',
                   'early_stopping_rounds': 10,
                   'num_class': 5} 

    # create learning curve for linear model
    learning_curve(model=model,
                   param_grid=param_grid,
                   df=df,
                   train_length = [3,6,12,24,48,96,240], 
                   train_end = to_datetime("2014-12-31"),
                   test_begin = to_datetime("2015-01-01"),
                   test_end = to_datetime("2017-10-31"),
                   file_surfix="xgboost")


    print("Successfully completed all tasks")


