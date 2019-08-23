import pandas as pd

#-------------------------------------------------------------------------------
# define functions
#-------------------------------------------------------------------------------
def count_null(df, columns):
    '''
    Calculate fraction of null values in given columns.
    Args:
        df: Pandas dataframe
        columns: columns of interest
    Return:
        p_null: fraction of null values in dictionary.
            Example: {"column1": 0.5, ...}
    '''
    p_null = {}
    for column in columns:
        p_null[column] = df[column].isnull().mean()
    return p_null

def winsorize_series(series, lower, upper):
    '''Winsorize each var by month
    Note: use pandas quantitle function instead of scipy.winsorize function,
    because of Scipy winsorize function has NaN handling bug,
    as noted in previous text
    Args:
        series: a Series
        lower, upper: winsorize threshold
    Return: winsorized series
    '''
    series = pd.Series(series)
    q = series.quantile([lower, upper])
    if isinstance(q, pd.Series) and len(q) == 2:
        series[series < q.iloc[0]] = q.iloc[0]
        series[series > q.iloc[1]] = q.iloc[1]    
    return series


def winsorize_df(df, features, lower, upper):
    ''' Winsorize given features
    Args:
        df: Pandas dataframe
        features: list of column names to winsorize
        lower, upper: winsorize threshold
    Return:
        dataframe with the given columns winsorized
    ''' 
    for feature in features:
        df[feature] = winsorize_series(df[feature], lower, upper)
    return df

def standardize_series(col):
    '''Normalize each column by month mean and std'''
    return (col - col.mean()) / col.std()

def standardize_df(df, features):
    '''Standardize dataframe by month mean and std'''
    for feature in features:
      df[feature] = standardize_series(df[feature])    
    return df


def remove_missing_targets(df, targets):
    '''the observations are removed if target variables are missing.
    The fraction of missing returns are printed.'''
    # check null values
    null_fraction = count_null(df, targets)
    for i, key in enumerate(null_fraction):
        io.message("Removing the observations with missing %s (%.4f)" \
            % (targets[i], null_fraction[key]))
    # remove null
    return df.dropna(subset=targets)  

def impute_data(df, method, features):
    ''' Impute missing data using the given imputation method. 
    Args:
        df: dataframe
        method: available options: month, securityId_ff, securityId_average
        features: features to impute.
    Return:
        imputed dataframe
    '''
    if method == "month":
        df = impute_by_month(df, features)
    elif method == "securityId_ff":
        df = impute_by_securityID(df, features)
    elif method == "securityId_average":
        df = impute_by_securityID_forward(df, features)
    else:
        io.message("Impute method is not valid.")
    return df

def impute_by_month(df, features):
    '''Impute missing data with the mean Z score within the same month group'''
    df[features] = df[features].fillna(0, inplace=False)
    return df



