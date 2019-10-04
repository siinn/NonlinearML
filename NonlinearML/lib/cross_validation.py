from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import NonlinearML.lib.io as io



def train_test_split_by_date(df, date_column, test_begin, test_end):
    ''' create train and test dataset by dates.
    Args:
        df: pandas dataframe
        date_column: date column in datetime format
        test_begin: test begin month in datetime format
        test_end: test end month in datetime format
    Return:
        df_train: train dataset
        df_test: test dataset
    '''
    # If dates are given in string, convert them into datetime
    if isinstance(test_begin, str):
        test_begin = datetime.strptime(test_begin, '%Y-%m-%d')
    if isinstance(test_end, str):
        test_end = datetime.strptime(test_end, '%Y-%m-%d')
    # Select train and test set
    df_test = df.loc[(df[date_column] >= test_begin) & (df[date_column] <= test_end)]
    df_train = df.loc[(df[date_column] < test_begin) | (df[date_column] > test_end)]
    return df_train, df_test


def create_purged_fold(
    df, val_begin, val_end, date_column, purge_length,
    embargo_length=0, subsample=1):
    """ Create purged training set and validation set as a one instance
    of K-folds.
    Args:
        df: Pandas dataframe
        val_begin, val_end: Begin and end date of validation period given
            in Datetime format.
        date_column: column representing time
        purge_length: Overlapping window size to be removed from
            training samples given in months.
            Example: The overlap between train and validation dataset of
            size (purge_length) will be removed from training samples.
        embargo_length: Training samples within the window of
            size (embargo_length) which follow the overlap between
            validation and train set will be removed. Embargo length is given
            in months.
        subsample: fraction of training samples to use.
    Return:
        df_purged_train: train and validation set for one instance of K-fold.
    """
    def _purge_train(
        df_train, val_begin, val_end, date_column,
        purge_length, embargo_length):
        """ Purge training set. Any overlapping samples and 'embargo' will be
        removed."""
        # Training samples before validation set to purge.
        # Given in tuple, (begin, end) dates
        overlap_before_val = (
            val_begin - relativedelta(months=purge_length), val_begin)
    
        # Training samples after validation set to purge.
        # Given in tuple, (begin, end) dates
        overlap_after_val = (
            val_end,
            val_end + relativedelta(months=purge_length)\
                    + relativedelta(months=embargo_length))
        # Get list of indices to drop
        index_before_val = df_train.loc[
            (df_train[date_column] >= overlap_before_val[0])\
            & (df_train[date_column] <= overlap_before_val[1])].index
        index_after_val = df_train.loc[
            (df_train[date_column] >= overlap_after_val[0])\
            & (df_train[date_column] <= overlap_after_val[1])].index
    
        # Return purged training set
        return df_train.drop(index_before_val.append(index_after_val))

    # Split train and validation
    df_train, df_val = train_test_split_by_date(
        df=df, date_column=date_column, test_begin=val_begin, test_end=val_end)

    # Purge training set by removing overlap and embargo
    # between training and validation sets.
    df_purged_train = _purge_train(
        df_train, val_begin, val_end, date_column, purge_length, embargo_length)

    # Subsample training set
    df_purged_train = df_purged_train.sample(frac=subsample)

    # Return purged training set
    return df_purged_train, df_val

def get_val_dates(df, k, date_column, verbose=False):
    ''' Find dates to be used to split data into K-folds.
    Args:
        df: Pandas dataframe.
        k: Number of partitions for K-folds.
        date_column: column representing time.
        verbose: Print debugging information if True
    Return:
        val_dates = list of (val_begin, val_end) tuples
    '''
    # Get begin and end dates
    date_begin = df[date_column].min().to_period('M')
    date_end = df[date_column].max().to_period('M')
    # Dictionary to hold results
    val_dates = []
    # Calculate the length of one fold
    fold_length = (date_end - date_begin).n / k

    # Warn that K-folds will not be of equal size
    if not isinstance(fold_length, int):
        # Remove decimals
        fold_length = int(fold_length)
        if verbose:
            io.message('Warning: K-folds will not be of equal size.')
            io.message('         K = %s, Fold size (month)= %s' %(k, fold_length))
            io.message('         date_begin = %s, date_end = %s' 
                %(date_begin, date_end))
    # Calculate begin and end dates of validation set.
    # Convert them back to timestamp.
    if verbose:
        io.message(
            'Dataset will be splitted into K-folds with the following dates:')
    for i in range(k):
        # Calculate validation begin and end dates
        val_begin = date_begin + i * fold_length
        val_end = date_begin + (i+1) * fold_length
        # Last fold will include the rest of dataset. Therefore,
        # use date_end instead of val_end
        if i+1 == k:
            if verbose:
                io.message(' > k = %s, begin = %s, end = %s, fold size (month) = %s'\
                %(i, val_begin, date_end, date_end-val_begin))
            val_dates.append(
                (val_begin.to_timestamp(), date_end.to_timestamp()))
        else:
            if verbose:
                io.message(' > k = %s, begin = %s, end = %s, fold size (month) = %s'\
                %(i, val_begin, val_end, val_end-val_begin))
            val_dates.append(
                ((val_begin.to_timestamp(), val_end.to_timestamp())))
    return val_dates





def evaluate_classifier(df, label_cla, y_pred, results):
    """ Evaluate classification model by calculating metrics.
    Args:
        df: Pandas dataframe containing prediction
        label_cla: Column name representing prediction
        y_pred: Prediction given in sequence
        results: Dictionary holding the evaluation results
    Return:
        results: updated results
    """
    # Get list of classes
    classes = sorted([str(x) for x in df[label_cla].unique()])
    report = classification_report(
        df[label_cla], y_pred, output_dict=True)
    # Append results
    for cls in classes:
        results['%s_f1-score' %cls].append(report[cls]['f1-score'])
        results['%s_precision' %cls].append(report[cls]['precision'])
        results['%s_recall' %cls].append(report[cls]['recall'])
    results['accuracy'].append(report['accuracy'])
    results['f1-score'].append(report['macro avg']['f1-score'])
    results['precision'].append(report['macro avg']['precision'])
    results['recall'].append(report['macro avg']['recall'])
    return results

def evaluate_regressor(y_true, y_pred, results):
    """ Calculate and return regression metrics, r2, MSE, and MAE."""
    results['r2'].append(r2_score(y_true, y_pred))
    results['mse'].append(mean_squared_error(y_true, y_pred))
    results['mae'].append(mean_absolute_error(y_true, y_pred))
    return results



def purged_k_fold_cv(
    df_train, model, model_type, features, label, metrics, k, purge_length, embargo_length,
    n_epoch=1, date_column='eom', subsample=1, verbose=False, rank=False, label_cla=False):
    """ Perform purged k-fold cross-validation. Assumes that data is uniformly
        distributed over the time period.
            i.e. Data is splitted by dates instead of size.
    Args:
        df_train: input Pandas dataframe
        model: Model with .fit(X, y) and .predict(X) method.
                features, label: List of features and target label
        model_type: either 'reg' or 'cla' for regression or claassification,
            respectively.
        features: List of features
        label: target label
        metrics: metric used to evaluate models such as ['r2'] for regression
            or ['f1-score'] for classification.
        k: k for k-fold CV.
        purge_length: Overlapping window size to be removed 
            from training samples given in months.
            i.e. the overlap between train and validation dataset
            of size (purge_length) will be removed from training samples.
        embargo_length: Training samples within the window of size
            (embargo_length) which follow the overlap between validation and
            train set will be removed. Embargo length is given in months.
        n_epoch: Number of times to repeat cross-validation.
        date_column: Datetime column
        subsample: fraction of training samples to use.
        verbose: Print debugging information if True
        rank: If True, prediction is made by ranking the regression output.
        label_cla: classification label. Ignored when model_type is 'reg'.
    Return:
        results[mean]: Dictionary containing average performance across
            k folds for each metric. ex. {'accuracy':0.3, 'f1-score':0.5, etc.}
        results[std]: Dictionary containing standard deviation of performance
            across k folds for each metric.
            ex. {'accuracy':0.3, 'f1-score':0.5, etc.}
        results: Raw cross-validation results. May used for plotting
            distribution. """
    io.message('Performing cross-validation with purged k-fold')
    io.message('\t> purge length = %s' % purge_length)
    io.message('\t> embargo length = %s' % embargo_length)
    # Find dates to be used to split data into k folds.
    val_dates = get_val_dates(
        df=df_train, k=k, date_column=date_column, verbose=verbose)
    # Dictionary to hold results
    results = {metric:[] for metric in metrics}
    # Loop over k folds, n_epoch times
    i = 0
    while i < n_epoch:
        i = i+1
        for val_begin, val_end in val_dates:
            # Print debugging info
            if verbose==True:
                io.message('Creating an instance of purged k-fold, epoch=%s' %(i//k))
                io.message('\t> validation begin = %s' % val_begin.to_period('M'))
                io.message('\t> validation end = %s' % val_end.to_period('M'))
            # Create purged training set and validation set as
            # a one instance of k folds.
            df_k_train, df_k_val = create_purged_fold(
                df=df_train,
                purge_length=purge_length,
                embargo_length=embargo_length,
                val_begin=val_begin,
                val_end=val_end,
                date_column=date_column,
                subsample=subsample)
            # Fit and make prediction
            model.fit(X=df_k_train[features], y=df_k_train[label])
            if rank:
                y_pred = model.predict(df_k_val[features], df_k_val[date_column])
            else:
                y_pred = model.predict(df_k_val[features])

            # Evaluate model
            if model_type=='cla':
                results = evaluate_classifier(df_k_val, label_cla, y_pred, results)
            if model_type=='reg':
                results = evaluate_regressor(df_k_val[label].values, y_pred, results)

    # Return results averaged over k folds
    results_mean = {metric:np.array(results[metric]).mean()
        for metric in results}
    results_std = {metric:np.array(results[metric]).std()
        for metric in results}
    if verbose==True:
        io.message("\t>> Validation performance:")
        for key in results_mean:
            io.message("\t\t>> mean %s = %s" % (key, results_mean[key]))
        for key in results_std:
            io.message("\t\t>> std %s = %s" % (key, results_std[key]))
    return {'mean':results_mean, 'std':results_std, 'values':results}




def get_classification_metrics(df, label_cla):
    """ Return classification metrics given dataframe and target label."""
    # Get list of classes
    classes = sorted([str(x) for x in df[label_cla].unique()])
    # List of all available metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    for cls in classes:
        metrics = metrics + [
            '%s_precision' %cls, '%s_recall' %cls, '%s_f1-score' %cls]
    return metrics



def grid_search(
    df_train, model, model_type, param_grid, features, label, k, purge_length,
    output_path, n_epoch=1, embargo_length=0, date_column='eom', subsample=1,
    verbose=False, label_cla=False, rank=False):
    ''' Perform grid search using purged cross-validation method. 
    Args:
        df_train: training set given in Pandas dataframe
        model: Model with .fit(X, y) and .predict(X) method.
        model_type: either 'reg' or 'cla' for regression or claassification,
            respectively.
        params_grid: Hyperparamater grid to search.
        features, label: List of features and target label
        k: k for k-fold CV.
        purge_length: Overlapping window size to be removed from training
            samples given in months.
            i.e. the overlap between train and validation dataset of size
            (purge_length) will be removed from training samples.
        embargo_length: Training samples within the window of size
            (embargo_length) which follow the overlap between validation
            and train set will be removed. Embargo length is given in months.
        date_column: Datetime column
        subsample: fraction of training samples to use.
        verbose: Print debugging information if True
        output_path: Path to save results as csv
        label_cla: classification label. Ignored when model_type is 'reg'.
        rank: If True, prediction is made by ranking the regression output.
    Return:
        cv_results: Dataframe summarizing cross-validation results
    '''
    io.title('Grid search with k-fold CV: k = %s, epoch = %s' % (k, n_epoch))
    if model_type=='cla':
        metrics = get_classification_metrics(df_train, label_cla)
    elif model_type=='reg':
        metrics = ['r2', 'mse', 'mae']
    else:
        io.error("Incorrect model type. Choose either 'reg' or 'cla'.")
    # Get all possible combination of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Define dataframe to hold results
    cv_results = pd.DataFrame(experiments)
    # Loop over different values of k
    n_experiments = len(experiments)
    count=0
    for i, params in enumerate(experiments):
        count = count + 1
        io.message("Experiment (%s/%s)" % (count, n_experiments))
        io.message("Parameters:")
        io.message(["\t - "+x+"="+str(params[x]) for x in params])
        # Perform purged k-fold cross validation
        single_model_result = purged_k_fold_cv(
            df_train=df_train,
            model=model.set_params(**params),
            model_type=model_type,
            features=features, label=label, metrics=metrics,
            date_column=date_column,
            k=k, verbose=verbose, n_epoch=n_epoch,
            purge_length=purge_length,
            embargo_length=embargo_length,
            subsample=subsample,
            label_cla=label_cla, rank=rank)
        # Save evaluation result of all metrics, not just one that is used.
        for m in metrics:
            cv_results.at[i, m] = single_model_result['mean'][m]
            cv_results.at[i, m+"_std"] = single_model_result['std'][m]
            cv_results.at[i, m+"_values"] = str(
                single_model_result['values'][m])
        # Save parameters as one string
        cv_results.at[i, 'params'] = str(
            [x+"="+str(params[x]) for x in params])\
            .strip('[]').replace('\'','')
    return cv_results




