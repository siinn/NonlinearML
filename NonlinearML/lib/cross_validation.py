import ast
from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score


import NonlinearML.lib.io as io
import NonlinearML.lib.utils as utils


def train_test_split_by_date(
    df, date_column, test_begin, test_end, train_begin=None, train_end=None,
    train_from_future=False, date_format='%Y-%m-%d'):
    ''' Create train and test dataset by dates. Test dataset includes test_begin
    dates.
    Args:
        df: pandas dataframe
        date_column: date column in datetime format
        test_begin, train_end: test begin and end month in datetime format
        train_begin, train_end: train begin and end month in datetime format
            if None, the rest of dataset is used as training set
        train_from_future: If True, train set can also include dates after
            validation set. Set to False if you want to avoid having training
            set dates come after validation set.
    Return:
        df_train: train dataset
        df_test: test dataset
    '''
    # If dates are given in string, convert them into datetime
    for date in [train_begin, train_end, test_begin, test_end]:
        if isinstance(date, str):
            date = datetime.strptime(date, date_format)
    # Select train and test set
    df_test = df.loc[\
        (df[date_column] >= test_begin)\
        & (df[date_column] <= test_end)]
    if train_begin and train_end:
        if train_from_future:
            df_train = df.loc[\
                (df[date_column] >= train_begin)\
                & (df[date_column] <= train_end)]
        # When creating training set, do not include dates after test_begin
        else:
            df_train = df.loc[\
                ((df[date_column] >= train_begin)\
                & (df[date_column] <= train_end)) & (df[date_column] < test_begin)]
    else: # If train_begin,end is not defined, use the rest as train set
        if train_from_future:
            df_train = df.loc[\
                (df[date_column] < test_begin)\
                | (df[date_column] > test_end)]
        # When creating training set, do not include dates after test_begin
        else: 
            df_train = df.loc[df[date_column] < test_begin]
    return df_train, df_test


def create_purged_fold(
    df, val_begin, val_end, date_column, purge_length,
    embargo_length=0, subsample=1, train_from_future=False, verbose=False):
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
        train_from_future: If True, train set can also include dates after
            validation set. Set to False if you want to avoid having training
            set dates come after validation set.
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
            val_begin - relativedelta(months=purge_length),
            val_begin - relativedelta(months=1))
    
        # Training samples after validation set to purge.
        # Given in tuple, (begin, end) dates
        overlap_after_val = (
            val_end + relativedelta(months=1),
            val_end + relativedelta(months=purge_length)\
                    + relativedelta(months=embargo_length))
        # Get list of indices to drop
        index_before_val = df_train.loc[
            (df_train[date_column] >= overlap_before_val[0])\
            & (df_train[date_column] <= overlap_before_val[1])].index
        index_after_val = df_train.loc[
            (df_train[date_column] >= overlap_after_val[0])\
            & (df_train[date_column] <= overlap_after_val[1])].index

        if verbose:
            def _to_date(timestamp):
                """ Return string in YEAR-MONTH format"""
                return "-".join([str(timestamp.year), str(timestamp.month)])
            io.message('Purging the samples within the following time period (Inclusive)')
            io.message('\t> Overlap before training set: %s - %s'
                % (_to_date(overlap_before_val[0]),
                    _to_date(overlap_before_val[1])))
            io.message('\t> Overlap after training set: %s - %s'
                % (_to_date(overlap_after_val[0]),
                    _to_date(overlap_after_val[1])))
    
        # Return purged training set
        return df_train.drop(index_before_val.append(index_after_val))

    # Split train and validation
    df_train, df_val = train_test_split_by_date(
        df=df, date_column=date_column, test_begin=val_begin, test_end=val_end,
        train_from_future=train_from_future)

    # Purge training set by removing overlap and embargo
    # between training and validation sets.
    df_purged_train = _purge_train(
        df_train, val_begin, val_end, date_column, purge_length, embargo_length)

    # Subsample training set
    df_purged_train = df_purged_train.sample(frac=subsample)

    # Return purged training set
    return df_purged_train, df_val

def get_val_dates(df, k, date_column, force_val_length=False, verbose=False):
    ''' Find dates to be used to split data into K-folds.
    Args:
        df: Pandas dataframe.
        k: Number of partitions for K-folds.
        date_column: column representing time.
        force_val_length: (boolean, int) If integer is given, validation set
            will have the length of the given integer. ex. 1 month
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
    if force_val_length:
        fold_length = force_val_length
    else:
        fold_length = ((date_end - date_begin).n / k) + 1

    # Warn that K-folds will not be of equal size
    if not isinstance(fold_length, int):
        # Remove decimals
        fold_length = int(fold_length)
        if verbose:
            io.message('Warning: K-folds will not be of equal size.')
            io.message('         K = %s, Fold size (month)= %s'%(k,fold_length))
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
        val_end = date_begin + (i+1) * fold_length - 1
        # Last fold will include the rest of dataset. Therefore,
        # use date_end instead of val_end
        if i+1 == k:
            if verbose:
                io.message(' > k = %s, begin = %s, end = %s, fold size (month) = %s'
                %(i, val_begin, date_end, date_end-val_begin))
            val_dates.append(
                (val_begin.to_timestamp(), date_end.to_timestamp()))
        else:
            if verbose:
                io.message(' > k = %s, begin = %s, end = %s, fold size (month) = %s'
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
        results['%s_f1-score' %cls] =\
            results.get('%s_f1-score' %cls, []) + [report[cls]['f1-score']]
        results['%s_precision' %cls] =\
            results.get('%s_precision' %cls, []) + [report[cls]['precision']]
        results['%s_recall' %cls] = \
            results.get('%s_recall' %cls, []) + [report[cls]['recall']]
    results['accuracy'] =\
        results.get('accuracy', []) + [report['accuracy']]
    results['f1-score'] =\
        results.get('f1-score', []) + [report['macro avg']['f1-score']]
    results['precision'] =\
        results.get('precision', []) + [report['macro avg']['precision']]
    results['recall'] = \
        results.get('recall', []) + [report['macro avg']['recall']]
    return results


def evaluate_regressor(y_true, y_pred, results, epsilon=1e-7):
    """ Calculate and return regression metrics, r2, MSE, MAE, MAPE, and MLSE.
    Model is selected by the highest score.
    i.e. These metrics should have higher score for better performing model.
    """
    results['r2'] = results.get('r2', []) + [r2_score(y_true, y_pred)]
    results['mse'] = results.get('mse', []) + [-1*pow((y_true-y_pred),2).mean()]
    results['mae'] = results.get('mae', []) + [-1*abs(y_true - y_pred).mean()]
    results['mape'] = results.get('mape', []) + [-1*(100 * abs(y_true - y_pred)\
            / np.maximum(abs(y_true), epsilon)).mean()]
    results['mlse'] =\
        results.get('mlse', []) + [-1*np.log1p(pow(y_pred-y_true,2)).mean()]
    return results

def evaluate_top_bottom_strategy(
    df, date_column, label_fm, y_pred,
    rank_n_bins, rank_order, rank_top, rank_bottom, results):
    """ Evaluate prediction by Top - Bottom strategy. In this evaluation,
    samples are ranked monthly, and average difference in target values
    (such as return) between top and bottom classes is calculated.
    Args:
        df: Dataframe containing date and target variables of k-fold
            validation set
        date_column: Datetime column
        label_fm: This represents return you wish to use for strategy.
        y_pred: Prediction from model
        rank_n_bins: Number of classes used in top-bottom strategy
            ex. 10 for decile, 5 for quintile, etc.
        rank_order: Class labels in ascending order.
            Example. [2, 1, 0] -> class 0 has highest values
        rank_top, rank_bottom: Top and bottom class. Must match with rank_order
            ex. rank_top = 0, rank_bottom = 9.
    Return:
        results: results including evaluation from top-bottom strategy
        
    """
    # Append prediction to dataframe. Assume that they are aligned.
    df['y_pred'] = y_pred
    # Discretize prediction
    df = utils.discretize_variables_by_month(
        df, variables=['y_pred'], n_classes=rank_n_bins,
        class_names=rank_order, suffix="discrete",
        month=date_column)
    # Calculate monthly average of each group
    df_monthly = df.groupby([date_column,'y_pred_discrete'])\
        .mean()[label_fm].reset_index()
    # Calculate difference between top and bottom 
    df_diff = df_monthly.loc[df_monthly['y_pred_discrete']==rank_top]\
        .reset_index()[label_fm]\
        - df_monthly.loc[df_monthly['y_pred_discrete']==rank_bottom]\
        .reset_index()[label_fm]
    results['Top-Bottom'] = results.get('Top-Bottom', []) + [df_diff.mean()]
    results['Top-Bottom-std'] = \
        results.get('Top-Bottom-std', []) + [df_diff.mean() / df_diff.std()]

    return results


def evaluate_model(
    model_type, df, label, label_fm, y_pred, results, date_column,
    rank_n_bins, rank_order, rank_top, rank_bottom):
    """ Evaluate trained model using pre-defined metrics and append it to
        results. 
            ex. {'r2':[0.01, 0.02], 'mse': [0.001, 0.003]}
        Each entry in list represents performance of single k-fold.
        i.e. Example above shows performance of k=2 cross-validation.
    Args: 
        model_type: either 'reg' or 'cla' for regression or claassification,
            respectively.
        df: Samples from which predictions are made.
            ex. train or validation set.
        label: target label
        label_fm: Only used for top-bottom strategy evaluation. This represents
            return you wish to use for strategy.
        y_pred: Prediction made by mode
        k: k for k-fold CV.
        date_column: Datetime column
        rank_n_bins, rank_order, rank_top, rank_bottom: If not None, 
            cross-validation is also evaluated based on Top-Bottom strategy.
    Return:
        results: Dictionary containing evaluation results
    """
    # Evaluate model
    if model_type=='cla':
        results = evaluate_classifier(
            df, label, y_pred, results)
    if model_type=='reg':
        results = evaluate_regressor(
            df[label].values, y_pred, results)
        # Evaluate regressor using Top-Bottom strategy
        if rank_n_bins and rank_order:
            results = evaluate_top_bottom_strategy(
                df=df, date_column=date_column,
                label_fm=label_fm, y_pred=y_pred, 
                rank_n_bins=rank_n_bins, rank_order=rank_order,
                rank_top=rank_top, rank_bottom=rank_bottom,
                results=results)
    return results

def purged_k_fold_cv(
    df_train, model, model_type, features, label, label_fm, k,
    purge_length, embargo_length, n_epoch=1, date_column='eom', subsample=1,
    rank_n_bins=None, rank_order=None, rank_top=None, rank_bottom=None,
    train_from_future=False, force_val_length=False, verbose=False):
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
        label_fm: Only used for top-bottom strategy evaluation. This represents
            return you wish to use for strategy.
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
        rank_n_bins, rank_order, rank_top, rank_bottom: If not None, 
            cross-validation is also evaluated based on Top-Bottom strategy.
        train_from_future: If True, train set can also include dates after
            validation set. Set to False if you want to avoid having training
            set dates come after validation set.
        force_val_length: (boolean, int) If integer is given, validation set
            will have the length of the given integer. ex. 1 month
        verbose: Print debugging information if True
    Return:
        results[mean]: Dictionary containing average performance across
            k folds for each metric. ex. {'accuracy':0.3, 'f1-score':0.5, etc.}
        results[std]: Dictionary containing standard deviation of performance
            across k folds for each metric.
            ex. {'accuracy':0.3, 'f1-score':0.5, etc.}
        results: Raw cross-validation results. May used for plotting
            distribution. """
    io.message('Performing cross-validation with purged k-fold')
    io.message('\t> Purge length = %s' % purge_length)
    io.message('\t> Embargo length = %s' % embargo_length)
    io.message('\t> Train from future = %s' % train_from_future)
    # Find dates to be used to split data into k folds.
    val_dates = get_val_dates(
        df=df_train, k=k, date_column=date_column,
        force_val_length=force_val_length, verbose=verbose)
    # Dictionary to hold results
    results_val = {}
    results_train = {}
    # Loop over k folds, n_epoch times
    i = 0
    while i < n_epoch:
        i = i+1
        for val_begin, val_end in val_dates:
            # Print debugging info
            if verbose==True:
                io.message('Creating an instance of purged k-fold, epoch=%s'
                    %i, newline=True)
                io.message('\t> validation begin = %s'
                    % val_begin.to_period('M'))
                io.message('\t> validation end = %s'
                    % val_end.to_period('M'))
                io.message('\t> Include future dataset in training: %s'
                    % train_from_future)
            # Create purged training set and validation set as
            # a one instance of k folds.
            df_k_train, df_k_val = create_purged_fold(
                df=df_train,
                purge_length=purge_length,
                embargo_length=embargo_length,
                val_begin=val_begin,
                val_end=val_end,
                date_column=date_column,
                subsample=subsample,
                train_from_future=train_from_future,
                verbose=verbose)

            # Skip if there is no training set
            if len(df_k_train)==0:
                io.message('Skipping the following k-fold as training set is empty')
                io.message('\t> validation begin = %s'
                    % val_begin.to_period('M'))
                io.message('\t> validation end = %s'
                    % val_end.to_period('M'))
                continue

            # Fit and make prediction
            model.fit(X=df_k_train[features], y=df_k_train[label])
            y_pred_val = model.predict(df_k_val[features])
            y_pred_train = model.predict(df_k_train[features])
            # Evaluate model
            results_train = evaluate_model(
                model_type, df_k_train, label, label_fm, y_pred_train,
                results_train, date_column,
                rank_n_bins, rank_order, rank_top, rank_bottom)

            results_val = evaluate_model(
                model_type, df_k_val, label, label_fm, y_pred_val, results_val,
                date_column, rank_n_bins, rank_order, rank_top, rank_bottom)

    # Return results averaged over k folds
    results_train_mean = {metric:np.array(results_train[metric]).mean()
        for metric in results_train}
    results_train_std = {metric:np.array(results_train[metric]).std()
        for metric in results_train}
    results_val_mean = {metric:np.array(results_val[metric]).mean()
        for metric in results_val}
    results_val_std = {metric:np.array(results_val[metric]).std()
        for metric in results_val}

    if verbose==True:
        io.message("\t>> Train performance:")
        for key in results_train_mean:
            io.message("\t\t>> mean %s = %s" % (key, results_train_mean[key]))
        for key in results_train_std:
            io.message("\t\t>> std %s = %s" % (key, results_train_std[key]))
        io.message("\t>> Validation performance:")
        for key in results_val_mean:
            io.message("\t\t>> mean %s = %s" % (key, results_val_mean[key]))
        for key in results_val_std:
            io.message("\t\t>> std %s = %s" % (key, results_val_std[key]))
    return {
        'train_mean':results_train_mean,
        'train_std':results_train_std,
        'train_values':results_train,
        'val_mean':results_val_mean,
        'val_std':results_val_std,
        'val_values':results_val
        }




def get_classification_metrics(df, label):
    """ Return classification metrics given dataframe and target label."""
    # Get list of classes
    classes = sorted([str(x) for x in df[label].unique()])
    # List of all available metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    for cls in classes:
        metrics = metrics + [
            '%s_precision' %cls, '%s_recall' %cls, '%s_f1-score' %cls]
    return metrics


def _paramsToString(params, param_grid):
    """ Convert dictionary of params to string.
    Args:
        params: current parameter set
        param_grid: dictionary containing entire parameter space. Used
            to get describtion if parameter if exists.
    """
    # Convert to list of key=val
    param_str = []
    for p in params:
        # If parameter is given as string, append them to output
        if type(param_grid[p]) == list:
            param_str.append('='.join([p, str(params[p])]))
        # If parameter is given as dictionary of {param:description},
        # get description instead.
        elif type(param_grid[p]) == dict:
            param_str.append('='.join([p, param_grid[p][params[p]]]))
    param_str = str(param_str).strip('[]').replace('\'','')
    return param_str

def _GetParamsDesc(params, param_grid):
    """ Get dictionary of {param: value or description}
    Args:
        params: current parameter set
        param_grid: dictionary containing entire parameter space. Used
            to get describtion if parameter if exists.
    """
    output = {}
    for p in params:
        # If parameter is given as dictionary of {param:description},
        # get description instead.
        if type(param_grid[p]) == dict:
            output[p] = '='.join([p, param_grid[p][params[p]]])
        # If parameter is given as string, append them to output
        else:
            output[p] = '='.join([p, str(params[p])])
    return output


def standardize_metrics(cv_results, cv_metric):
    """ Standardize metrics by mean and std of all models across k-folds
    Args:
        cv_results: Dataframe summarizing cross-validation results
        cv_metric: List of metric to use in calculating combined metric
    Return:
        cv_results: Same dataframe with additional columns representing
            standardized metrics.
    """
    for metric in cv_metric:
        for partition in ['train', 'val']:
            # Convert columns of list to multiple columns
            df_values = utils\
                .expand_column(cv_results, metric+"_%s_values" %partition)
            # Standardize by mean and std calculated from all models
            mean = df_values.stack().mean()
            std = df_values.stack().std()
            cv_results[metric+"_zscore_%s_mean" %partition] = mean
            cv_results[metric+"_zscore_%s_std" %partition] = std
            # Convert them to strings for compatibility
            z_score = df_values.apply(lambda x:(x-mean)/std).values.tolist()
            cv_results[metric+"_zscore_%s_values" %partition] = z_score
            cv_results[metric+"_zscore_%s_values" %partition] = \
                cv_results[metric+"_zscore_%s_values" %partition]\
                    .apply(lambda x:str(x))
    return cv_results

def combined_metrics(cv_results, cv_metric):
    """ Calculate combined metrics from standardize metrics.
    Args:
        cv_results: Dataframe summarizing cross-validation results
        cv_metric: List of metric to use in calculating combined metric
    Return:
        cv_results: Same dataframe with additional column representing
            combined metric.
    """
    # First standardize other metrics
    standardized_metrics = standardize_metrics(cv_results, cv_metric)
    for partition in ['train', 'val']:
        # Store z-score of each metrics.
        zscores = [] # ex. [z-score of r2, z-score of MSE, etc.]
        for metric in cv_metric:
            # Expand a single column to multiple columns
            df_values = utils\
                .expand_column(
                    standardized_metrics,
                    metric+"_zscore_%s_values" %partition)
            # Mean and std calculated from all models
            mean = df_values.stack().mean()
            std = df_values.stack().std()
            zscores.append(df_values.apply(lambda x:(x-mean)/std))
        # Calculate equal-weighted combined metric
        combined_metric = (sum(zscores) / len(zscores))
        # Calculate mean and std of the combined metric
        cv_results['combined_zscore_%s_mean' %partition] = \
            (sum(zscores) / len(zscores)).mean(axis=1)
        cv_results['combined_zscore_%s_std' %partition] = \
            (sum(zscores) / len(zscores)).std(axis=1)
        cv_results['combined_zscore_%s_values' %partition] = \
            (sum(zscores) / len(zscores)).values.tolist()
        cv_results['combined_zscore_%s_values' %partition] = \
            cv_results['combined_zscore_%s_values' %partition]\
                .apply(lambda x:str(x))
    return cv_results

def grid_search(
    df_train, model, model_type, param_grid, features, label, label_fm,
    k, purge_length, output_path, cv_metric, n_epoch=1, embargo_length=0,
    date_column='eom', subsample=1, train_from_future=False,
    rank_n_bins=None, rank_order=None, rank_top=None, rank_bottom=None,
    force_val_length=False, verbose=False):
    ''' Perform grid search using purged cross-validation method. 
    Args:
        df_train: training set given in Pandas dataframe
        model: Model with .fit(X, y) and .predict(X) method.
        model_type: either 'reg' or 'cla' for regression or claassification,
            respectively.
        params_grid: Hyperparamater grid to search.
        features, label: List of features and target label
        label_fm: Only used for top-bottom strategy evaluation. 
        k: k for k-fold CV.
        purge_length: Overlapping window size to be removed from training
            samples given in months.
            i.e. the overlap between train and validation dataset of size
            (purge_length) will be removed from training samples.
        output_path: Path to save results as csv
        cv_metric: List of metric to use in calculating combined metric
        n_epoch: Number of times to repeat CV
        embargo_length: Training samples within the window of size
            (embargo_length) which follow the overlap between validation
            and train set will be removed. Embargo length is given in months.
        date_column: Datetime column
        subsample: fraction of training samples to use.
        rank_n_bins, rank_order, rank_top, rank_bottom: If not None, 
            cross-validation is also evaluated based on Top-Bottom strategy.
        force_val_length: (boolean, int) If integer is given, validation set
            will have the length of the given integer. ex. 1 month
        verbose: Print debugging information if True
    Return:
        cv_results: Dataframe summarizing cross-validation results
    '''
    io.title('Grid search with k-fold CV: k = %s, epoch = %s, subsample = %s' % (k, n_epoch, subsample))
    # Get all possible combination of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Define dataframe to hold results
    cv_results = pd.DataFrame()
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
            features=features, label=label, #metrics=metrics,
            label_fm=label_fm,
            date_column=date_column,
            k=k, verbose=verbose, n_epoch=n_epoch,
            purge_length=purge_length,
            embargo_length=embargo_length,
            subsample=subsample,
            train_from_future=train_from_future,
            rank_n_bins=rank_n_bins, rank_order=rank_order,
            rank_top=rank_top, rank_bottom=rank_bottom)
        # Save evaluation result of all metrics
        _params = _GetParamsDesc(params, param_grid)
        cv_results = cv_results.append(_params, ignore_index=True)
        for m in single_model_result['val_mean'].keys(): # m = metric
            cv_results.at[i, m+"_train_mean"] = single_model_result['train_mean'][m]
            cv_results.at[i, m+"_train_std"] = single_model_result['train_std'][m]
            cv_results.at[i, m+"_train_values"] = str(
                single_model_result['train_values'][m])
            cv_results.at[i, m+"_val_mean"] = single_model_result['val_mean'][m]
            cv_results.at[i, m+"_val_std"] = single_model_result['val_std'][m]
            cv_results.at[i, m+"_val_values"] = str(
                single_model_result['val_values'][m])
        # Save parameters as one string
        cv_results.at[i, 'params'] = _paramsToString(params, param_grid)
    # Calculate combined metric. Only possible when there are multiple CV results.
    if len(cv_results) > 1:
        cv_results = combined_metrics(cv_results, cv_metric)
    return cv_results



