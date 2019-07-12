from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import rrule
import dateutil
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from Asset_growth.lib.backtest import *

def to_datetime(date, date_format='%Y-%m-%d'):
    ''' convert given string to datetime"
    Args:
        date: date given in "YYYY-MM-DD" format
    Return:
        date in datetime format
    '''
    return datetime.strptime(date, date_format)

def get_tertile_boundary(df, variables):
    ''' get boundaries of tertile groups by calculating the average of the boundaries of two adjacent tertile groups.
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

def discretize_variables_by_month(df, variables, month="eom", labels_tertile={}, labels_quintile={}):
    ''' discretize variables by assigning a quintile and tertile class within each month. 
    Args:
        df: Pandas dataframe containing variables
        variables: list of variables to discretize
        month: column representing time
    Return:
        df: Pandas dataframe with columns named x_quintile, x_tertile for all variable x.
    '''
    # create classification labels
    for var in variables:
        # set labels
        if var in labels_tertile:
            lt=labels_tertile[var]
        else:
            lt=[var+" low", var+" mid", var+" high"]
        if var in labels_quintile:
            lq=labels_quintile[var]
        else:
            lq=[var+" low", var+" mid-low", var+" mid", var+" mid-high", var+" high"]
        # assign classes
        df[var+"_tertile"] = df.groupby([month])[var].transform(lambda x: pd.qcut(x, 3, lt))
        df[var+"_quintile"] = df.groupby([month])[var].transform(lambda x: pd.qcut(x, 5, lq))
    return df




def make_prediction(model, df_train, df_test, features, label):
    ''' Train model with df_train and make predictions on df_test
    Args:
        df_train, df_test: train and test dataset in Pandas DataFrame
        features: list of features
        label: name of label column
    Return:
        pred_test: prediction of test dataset.
    '''
    # Fit model
    model.fit(X=df_train[features], y=df_train[label])
    # Make prediction
    pred_test = model.predict(df_test[features])
    return pred_test
    


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


def predict_and_calculate_cum_return(model, df_train, df_test, features, label_cla, label_fm, time="eom"):
    ''' Make prediction using the fitted model. Then calculate cumulative return using the prediction.
    Args:
        model: sklearn model that supports .fit and .predict method
        df_train, df_test: train and test dataframe
        features: list of features
        label_cla: name of column in dataframe that represent classification label
        label_fm: name of column in dataframe used for calculating cumulative return
        time: column name representing time
    Return:
        df_cum_return_train: cumulative return calculated from train dataset
        df_cum_return_test: cumulative return calculated from test dataset
        model: trained model
    '''
    # Fit model
    print("Fitting model..")
    print(model)
    model.fit(df_train[features], df_train[label_cla])
    # Concatenate prediction and true label
    pred_train  = concat_pred_label(df=df_train,
                                    prediction=model.predict(df_train[features]),
                                    columns=[time, label_cla, label_fm])
    pred_test  = concat_pred_label(df=df_test,
                                    prediction=model.predict(df_test[features]),
                                    columns=[time, label_cla, label_fm])
    # Calculate cumulative return
    df_cum_return_train = cumulative_return_from_classification(pred_train, var="pred", var_classes="pred", total_return=label_fm, time=time)
    df_cum_return_test = cumulative_return_from_classification(pred_test, var="pred", var_classes="pred", total_return=label_fm, time=time)
    return df_cum_return_train, df_cum_return_test, model


def grid_search(model, param_grid, df_train, df_val, features, label_cla, label_fm):
    ''' Perform grid search and return the best parameter set based on the following metric:
        metric: val_diff - abs(train_diff - val_diff)
            where train_diff = difference in cumulative return (Q1-Q3) in training set 
                  val_diff = difference in cumulative return (Q1-Q3) in validation set 
    Args:
        model: sklearn model that supports .fit and .predict method
        df_train, df_val: train and validation dataframe
        features: list of features
        label_cla: name of column in dataframe that represent classification label
        label_fm: name of column in dataframe used for calculating cumulative return
    Return:
        (best parameters, summary in dataframe)
    '''
    print("Performing grid search using validation set")
    # Dictionary to hold results
    summary = {"params":[], "train_top":[], "train_bottom":[], "train_diff":[], "val_top":[], "val_bottom":[], "val_diff":[]}
    # Get all possible combination of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Loop over different values of k
    n_experiments = len(experiments)
    count=0
    for params in experiments:
        count = count + 1
        print("---------------------------")
        print("Experiment (%s/%s)" % (count, n_experiments))
        print("---------------------------")
        print("Parameters:")
        print(params)
        # Calculate cumulative return after training model
        df_cum_train, df_cum_val, model = predict_and_calculate_cum_return(model=model.set_params(**params),
                                                             df_train=df_train, df_test=df_val,
                                                             features=features, label_cla=label_cla, label_fm=label_fm)
        # Calculate difference in cumulative return
        return_train_top, return_train_bottom, return_train_diff = last_cum_return(df_cum_train)
        return_val_top, return_val_bottom, return_val_diff = last_cum_return(df_cum_val)
        print("Cumulative return")
        print(" > Train")
        print("  >> top: %s" % return_train_top)
        print("  >> bottom: %s" % return_train_bottom)
        print("  >> diff: %s" % return_train_diff)
        print(" > Validation")
        print("  >> top: %s" % return_val_top)
        print("  >> bottom: %s" % return_val_bottom)
        print("  >> diff: %s" % return_val_diff)
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
    #df_return["metric"] = df_return["val_diff"] / abs(df_return["train_diff"] - df_return["val_diff"])
    df_return["metric"] = df_return["val_diff"] - abs(df_return["train_diff"] - df_return["val_diff"])
    # Return best parameter
    best_params = df_return.iloc[df_return["metric"].idxmax()]["params"]
    print("Best parameters:")
    print(best_params)
    return (best_params, df_return)



def grid_search_cv(model, param_grid, df_train, features, label_cla, average='macro', n_folds=5):
    ''' Perform grid search and return fitted GridSearchCV object.
    Args:
        model: sklearn model that supports .fit and .predict method
        param_grid: parameter grid to search
        df_train: train dataset
        features: list of features
        label_cla: name of column in dataframe that represent classification label
        average: average method for multiclass classification metric
        n_fold: number of CV folds
    Return:
        cv: fitted GridSearchCV object
    '''
    # custom scorer for multi-class classification
    scorer = make_scorer(f1_score, average=average)
    # run grid search
    print("Performing grid search..") 
    cv = GridSearchCV(model, param_grid, cv=n_folds, scoring=scorer, refit=True, n_jobs=-1, verbose=1)
    cv.fit(df_train[features], df_train[label_cla])
    # Print best parameters
    print("Best parameters:")
    print(cv.best_params_)
    return cv


def save_summary(df_train, output_path, cv_results):
    '''Collect results from all models trained with best parameters and save them as csv
    Args:
        df_train: Training dataset used to obtain class names (ex. 0.0, 1.0, 2.0)
        output_path: Path to save results
        cv_results: Results obtained from grid search using purged cross-validation. 
                    This is output of grid_search_purged_cv.
    Return:
        None
    '''
    # Get list of classes
    classes = [str(x) for x in df_train[label].unique()]
    # List of all available metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    for cls in classes:
        metrics = metrics + ['%s_precision' %cls, '%s_recall' %cls, '%s_f1-score' %cls]
    # Collect results from best parameters
    df_summary = pd.DataFrame({model:cv_results[model].loc[cv_results[model][metric].idxmax()][metrics]
                               for model in cv_results}).T
    # Save the summary
    df_summary.to_csv(output_path+'summary.csv')

