from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import rrule
import dateutil
import itertools
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as sklearn_train_test_split

import NonlinearML.lib.io as io

def create_folder(filename):
    """ Creates folder if not exists."""
    path = "/".join(filename.split('/')[:-1])
    if not os.path.exists(path):
        io.message("Creating folder: %s" %path)
        os.makedirs(path)
    return

def to_datetime(date, date_format='%Y-%m-%d'):
    ''' convert given string to datetime"
    Args:
        date: date given in "YYYY-MM-DD" format
    Return:
        date in datetime format
    '''
    return datetime.strptime(date, date_format)

def get_tertile_boundary(df, variables):
    ''' get boundaries of tertile groups by calculating the average of the
    boundaries of two adjacent tertile groups.
    Only works for tertile.
    Args:
        df: Pandas dataframe containing variables
        variables: list of variables of interests
    Return:
        results: dictionary containing boundaries of each variable
    '''
    results = {}
    for var in variables:
        lower_boundary = df.groupby("%s_tertile" %var).min()\
                           .sort_values(var, ascending=False)\
                           .head(2).sort_values(var)\
                           [var].values
        upper_boundary = df.groupby("%s_tertile" %var).max()\
                           .sort_values(var)\
                           .head(2)\
                           [var].values
        results[var] = list((lower_boundary + upper_boundary) / 2)
    return results

def train_test_split(df, date_column, test_begin, test_end, train_length=None, train_end=None):
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
    # Select test dataset
    df_test = df.loc[(df[date_column] >= test_begin) & (df[date_column] <= test_end)]

    # If train_length is not defined, select the rest as train dataset
    if train_length==None:
        df_train = df.loc[(df[time_column] < validation_begin) | (df[time_column] > validation_end)]
    else:
    	# If train_length is defined, only select data between train_end and (train_end - train_length)
        train_begin = train_end - dateutil.relativedelta.relativedelta(months=train_length)
        df_train = df.loc[(df[date_column] >= train_begin) & (df[date_column] <= train_end)]
    return df_train, df_test


def train_val_split_by_col(df, col, train_size=0.8):
    ''' Split train dataset into train and validation set using unique values of the given column.
    Args:
        df: pandas dataframe
        col: namem of column used for splitting. ex. "SecurityID"
        train_size: fraction of train set
    Return:
        df_train: train dataset
        df_val: val dataset
    '''
    def _split(x, train_size=train_size):
        ''' Split list into two groups by train_size'''
        train, val = sklearn_train_test_split(x, train_size=train_size)
        return train, val
    # Split unique values of column into train and validation
    df_securityID_by_sector = df.groupby("GICSSubIndustryNumber")["SecurityID"].unique()
    # Apply split function and get stratified list
    list_train = df_securityID_by_sector.apply(_split).apply(lambda x:x[0]).apply(pd.Series).stack().values
    list_val = df_securityID_by_sector.apply(_split).apply(lambda x:x[1]).apply(pd.Series).stack().values
    #list_train, list_val = sklearn_train_test_split(df[col].unique(), train_size=train_size) 
    # Create train and validation dataset using the sampled lists
    df_train = df.loc[df[col].isin(list_train)]
    df_val = df.loc[df[col].isin(list_val)]
    return df_train, df_val



def concat_pred_label(df, prediction, columns=[], pred_name='pred'):
    ''' Concatenate prediction, true classification label and numerical label into the same dataframe.
    Args:
        df: dataframe used to in prediction
        pred: prediction made by model
        columns: columns to copy from original dataframe. ex. ["eom", label_cla, label_reg]
        pred_name: name of prediction column
    '''
    # Combine prediction with true label
    df_result = pd.concat([df[columns].copy().reset_index().drop("index", axis=1),
                           pd.DataFrame(prediction, columns=['pred'])], axis=1)
    return df_result


def discretize_variables_by_month(
    df, variables, n_classes, class_names, suffix="discrete", month="eom"):
    ''' Discretize variables by assigning a class within each month. 
    Args:
        df: Pandas dataframe containing variables
        variables: list of variables to discretize
        n_classes: number of classes
        class_name: class labels in ascending order.
            Example. [2, 1, 0] -> class 0 is the top tercile
        suffix: suffix added to newly created column
        month: column representing time
    Return:
        df: Pandas dataframe with discretized variable.
    '''
    # Loop over each variable
    for var in variables:
        # Assign classes
        df["_".join([var, suffix])] = df.groupby([month])[var]\
            .transform(
                lambda x: pd.qcut(x.rank(method='first'), n_classes, class_names))
    return df


#def make_prediction(model, df_train, df_test, features, label):
#    ''' Train model with df_train and make predictions on df_test
#    Args:
#        df_train, df_test: train and test dataset in Pandas DataFrame
#        features: list of features
#        label: name of label column
#    Return:
#        pred_test: prediction of test dataset.
#    '''
#    # Fit model
#    model.fit(X=df_train[features], y=df_train[label])
#    # Make prediction
#    pred_test = model.predict(df_test[features])
#    return pred_test
    


def last_cum_return(df, time="eom"):
    ''' Return last cumulative return and the difference between top and bottom classes as below.
        diff = Q1 - Q3
    Args:
        df: dataframe containing cumulative return. output of fit_and_calculate_cum_return
        time: column representing time
    '''
    # Find last month
    last=df.sort_values(time, ascending=False)[time].iloc[0]
    df_last=df.loc[df[time]==last]
    # Sum return of top groups
    #return_top = 0
    #for top in sorted(df["pred"].unique())[:-1]:
    #    return_top = return_top + df_last.loc[df_last["pred"] == top]["cumulative_return"].iloc[0]
    # Calculate difference in return between top and bottom group
    return_top = df_last.loc[df_last["pred"] == sorted(df["pred"].unique())[0]]["cumulative_return"].iloc[0]
    return_bottom = df_last.loc[df_last["pred"] == sorted(df["pred"].unique())[-1]]["cumulative_return"].iloc[0]
    return_diff = return_top - return_bottom
    return return_top, return_bottom, return_diff


def predict(
    model, df_train, df_test, features, label, date_column, cols, rank=False):
    ''' Train model using best params and make prediction using trained model
    on both train and test dataset. label_fm which represent continuous target
    variable is joined to the prediction.
    Args:
        model: ML Model that supports .fit and .predict method
        df_train, df_test: train and test dataframe
        features: list of features
        label: target label
        cols: Other columns to include in output dataframe
        rank: If True, prediction is made by ranking the regression output.
    Return:
        pred_train, pred_test: prediction of train and test dataset
        model: trained model
    '''
    io.message("Making prediction:\n%s" % model)
    # Fit model
    model.fit(df_train[features], df_train[label])
    if rank:
        # Make prediction and concatenate prediction and true label
        pred_train  = concat_pred_label(
            df=df_train,
            prediction=model.predict(df_train[features], df_train[date_column]),
            columns=[date_column]+features+cols)
        pred_test  = concat_pred_label(
            df=df_test,
            prediction=model.predict(df_test[features], df_test[date_column]),
            columns=[date_column]+features+cols)
    else:
        # Make prediction and concatenate prediction and true label
        pred_train  = concat_pred_label(
            df=df_train,
            prediction=model.predict(df_train[features]),
            columns=[date_column]+features+cols)
        pred_test  = concat_pred_label(
            df=df_test,
            prediction=model.predict(df_test[features]),
            columns=[date_column]+features+cols)

    return pred_train, pred_test, model

#def calculate_cum_return(pred_train, pred_test, label_fm, time="eom"):
#    ''' Calculate cumulative return using the prediction.
#    Args:
#        pred_train, pred_test: prediction of train and test dataset. This is
#            an output of utils.prediction function.
#        label_fm: monthly return used for calculating cumulative return
#        time: column name representing time
#    Return:
#        df_cum_return_train: cumulative return calculated from train dataset
#        df_cum_return_test: cumulative return calculated from test dataset
#        model: trained model
#    '''
#    # Calculate cumulative return
#    io.title("Calculating cumulative return")
#    df_cum_return_train = cumulative_return_from_classification(
#        pred_train, var="pred", var_classes="pred",
#        total_return=label_fm, time=time)
#    df_cum_return_test = cumulative_return_from_classification(
#        pred_test, var="pred", var_classes="pred",
#        total_return=label_fm, time=time)
#    return df_cum_return_train, df_cum_return_test


def grid_search(
    model, param_grid, df_train, df_val, features, label_cla, label_fm, label_reg=False):
    ''' Perform grid search and return the best parameter set based on
    the following metric:
        metric: val_diff - abs(train_diff - val_diff)
            where train_diff = difference in cumulative return (Q1-Q3)
                in training set 
            val_diff = difference in cumulative return (Q1-Q3)
                in validation set 
    Args:
        model: sklearn model that supports .fit and .predict method
        df_train, df_val: train and validation dataframe
        features: list of features
        label_cla: name of column in dataframe that represent classification
            label
        label_fm: name of column used for calculating cumulative return
        label_reg: If specified, model is triained on this regression label. label_cla
            is ignored
    Return:
        (best parameters, summary in dataframe)
    '''
    io.message("Performing grid search using validation set")
    # Dictionary to hold results
    summary = {
        "params":[], "train_top":[], "train_bottom":[], "train_diff":[],
        "val_top":[], "val_bottom":[], "val_diff":[]}
    # Get all possible combination of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Loop over different values of k
    n_experiments = len(experiments)
    count=0
    for params in experiments:
        count = count + 1
        io.message("-----------------------------------------------------------")
        io.message("Experiment (%s/%s)" % (count, n_experiments))
        io.message("-----------------------------------------------------------")
        io.message("Parameters:")
        io.message(params)
        # Make prediction using best parameters
        pred_train, pred_test, model = utils.predict(
                model=model.set_params(**best_params),
                df_train=df_train, df_test=df_test, features=features,
                label_cla=label, label_fm=label_fm, time=date_column,
                label_reg=label_reg)
        # Calculate cumulative return using trained model
        df_cum_train, df_cum_test = utils.calculate_cum_return(
                pred_train=pred_train, pred_test=pred_test,
                label_fm=label_fm, time=date_column)
        # Calculate difference in cumulative return
        return_train_top, return_train_bottom, return_train_diff = \
            last_cum_return(df_cum_train)
        return_val_top, return_val_bottom, return_val_diff = \
            last_cum_return(df_cum_val)
        io.message("Cumulative return")
        io.message(" > Train")
        io.message("  >> top: %s" % return_train_top)
        io.message("  >> bottom: %s" % return_train_bottom)
        io.message("  >> diff: %s" % return_train_diff)
        io.message(" > Validation")
        io.message("  >> top: %s" % return_val_top)
        io.message("  >> bottom: %s" % return_val_bottom)
        io.message("  >> diff: %s" % return_val_diff)
        # Return k and difference
        summary["params"].append(params)
        summary["train_top"].append(return_train_top)
        summary["train_bottom"].append(return_train_bottom)
        summary["train_diff"].append(return_train_diff)
        summary["val_top"].append(return_val_top)
        summary["val_bottom"].append(return_val_bottom)
        summary["val_diff"].append(return_val_diff)
    # Convert results to dataframe
    df_return = pd.DataFrame(summary)
    df_return["metric"] = df_return["val_diff"] \
        - abs(df_return["train_diff"] - df_return["val_diff"])
    # Return best parameter
    best_params = df_return.iloc[df_return["metric"].idxmax()]["params"]
    io.message("Best parameters:")
    io.message(best_params)
    return (best_params, df_return)



def grid_search_cv(
    model, param_grid, df_train, features, label_cla, average='macro',
    n_folds=5):
    ''' Perform grid search and return fitted GridSearchCV object.
    Args:
        model: sklearn model that supports .fit and .predict method
        param_grid: parameter grid to search
        df_train: train dataset
        features: list of features
        label_cla: name of column in dataframe that represent
            classification label
        average: average method for multiclass classification metric
        n_fold: number of CV folds
    Return:
        cv: fitted GridSearchCV object
    '''
    # custom scorer for multi-class classification
    scorer = make_scorer(f1_score, average=average)
    # run grid search
    io.message("Performing grid search..") 
    cv = GridSearchCV(
        model, param_grid, cv=n_folds, scoring=scorer, refit=True,
        n_jobs=-1, verbose=1)
    cv.fit(df_train[features], df_train[label_cla])
    # Print best parameters
    io.message("Best parameters:")
    io.message(cv.best_params_)
    return cv

def get_param_string(params):
    """ Get name as a string."""
    names = []
    for key in params:
        if 'name' in dir(params[key]):
            names.append(key+'='+params[key].name)
        else:
            names.append(key+'='+str(params[key]))
        """ Todo: write a function to extract layer info
            if 'layers' in dir(params[key]):"""
    return ",".join(names)


def expand_column(df, col):
    """ Expand columns of strings into multiple columns.
    Ex. '['0.1', '0.4']' to two column of 0.1 0.4
    Args:
        df: Pandas dataframe
        col: Column to expand
    Return:
        Dataframe with multiple columns
    """
    return pd.DataFrame(df[col].str.strip('[]\' ')\
            .apply(lambda x:x.split(',') if type(x) == str else x)\
        .tolist())\
        .applymap(lambda x:float(x.strip(' \'')) if type(x) == str else float(x))



def rank_prediction_monthly(pred_train, pred_test, config, col_pred="pred"):
    """ Rank prediction within each month.
    Args:
	pred_train, pred_test: Dataframe containing prediction column
	config: Dictionary containing n_classes, class_order,
	    and date_column
    Return:
	pred_train, pred_test: Dataframe with additional column
	    representing ranks
    """
    pred_train = discretize_variables_by_month(
	df=pred_train,
	variables=[col_pred],
	n_classes=config['n_classes'],
	class_names=config['class_order'][::-1],
	suffix="rank", month=config['date_column'])

    pred_test = discretize_variables_by_month(
	df=pred_test,
	variables=[col_pred],
	n_classes=config['n_classes'],
	class_names=config['class_order'][::-1],
	suffix="rank", month=config['date_column'])
    return pred_train, pred_test

