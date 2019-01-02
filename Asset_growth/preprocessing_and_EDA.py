#!/usr/bin/env python
# import common python libraries
from __future__ import division
import numpy as np
import math
import matplotlib; matplotlib.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import pandas as pd
import seaborn as sns

# initial pyspark on local machine
import findspark
findspark.init()

# import pyspark libraries
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler, QuantileDiscretizer
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit, to_timestamp, array, udf
from pyspark.sql.functions import max as col_max
from pyspark.sql.types import StructField, StructType, IntegerType, DoubleType, StringType


#----------------------------------------------
# set user options
#----------------------------------------------

# set input and output path
input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r2.csv'
output_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context_r3.csv'
# set True for development
debug = False

# select algorithm to run    
perform_eda             = True
run_preprocessing       = True
impute_test             = True
impute_data             = True
feature_engineering     = True
create_label            = True
check_processed_data    = True

# set winsorization alpha
winsorize_alpha_lower = 0.05
winsorize_alpha_upper = 0.95    


# set plot style
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#plt.style.use('seaborn-talk')
plt.style.use('fivethirtyeight')


#----------------------------------------------
# define functions
#----------------------------------------------

if False:
    '''
    This needs to be performed only once. The original dataset is updated with an additional feature, FCFA.
    Also CVROIC is updated with more data as the original dataset was missing data.
    Missing data in CVROIC reduced from ~27% to ~13%.
    '''
    # set input paths
    input_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.csv'
    update_path = '/mnt/mainblob/asset_growth/data/20181105 New CVROIC.csv'
    output_path = '/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context.r2.csv'
    # import raw csv into spark dataframe
    df = spark.read.csv(input_path, header=True, nanValue=float('nan'), inferSchema=True)
    df_update = spark.read.csv(update_path, header=True, inferSchema=True)
    # replace NULL with python None
    df = df.na.replace("NULL", None)
    # join two dataframes
    df = df.join(df_update, ["SecurityID", "eom"]).drop(df.CVROIC)
    # save as new csv
    df.write.csv(output_path, header=True, mode="overwrite")

def count_null(df, columns):
    '''
    Calculate fraction of null values in given columns.
    Args:
        df: Spark dataframe
        columns: columns of interest
    Return:
        p_null: fraction of null values in dictionary. ex. {"column1": 0.5, ...}
    '''
    n_total = df.count()
    n_null = {}
    for column in columns:
        n_null[column] = df.filter(col(column).isNull()).count()
    p_null = {key:(float(value) / n_total) for key, value in n_null.items()}
    return p_null




#----------------------------------------------
# define plotting functions
#----------------------------------------------

def plot_dist_features(df, columns, n_rows=5, n_columns=3, n_bins=50, figsize=(10,10), log=True):
    '''
    Plot distribution of all given columns.
    Args:
        df: Spark dataframe
        columns: columns to plot
        others: plotting optoins
    Return: None
    '''
    # drop all rows containing null value for EDA purpose
    df_dropna = df.dropna("any")
    # create figure and axes
    fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize)
    ax.flatten()
    # loop over each columns
    for i, feature in enumerate(df_dropna.columns):
        # get row and column index given integer
        row = get_matrix_index(i, n_columns)[0]
        column = get_matrix_index(i, n_columns)[1]
        # get values of each column
        values = np.array(df_dropna.select(feature).collect())
        ax[row][column].hist(values, bins=n_bins, range=(values.min(), values.max()))
        # customize axes
        ax[row][column].set_xlabel(feature)
        ax[row][column].set_ylabel("Entries")
        ax[row][column].set_xticks(ax[row][column].get_xticks()[::2])
        if log:
            ax[row][column].set_yscale("log")
    # customize plot and save
    plt.tight_layout()
    if log:
        plt.savefig('plots/dist_all_features_log.png')
    else:
        plt.savefig('plots/dist_all_features_linear.png')
    return

def plot_null(df, columns, figsize=(15,5)):
    '''
    Plot percentage of null values for each of given columns.
    Args:
        df: Spark dataframe
        columns: columns of interest
        others: plotting optoins
    Return: None
    '''
    # create figure and axes
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # get fraction of null values
    p_null = count_null(df, df.columns)
    pdf = pd.DataFrame.from_dict(p_null, orient='index')
    # make bar plot
    pdf.plot.bar(y=0, ax=ax, legend=False)
    # annotate numbers
    y_offset = 0.01
    for p in ax.patches:
        bar = p.get_bbox()
        val = "{:+.3f}".format(bar.y1 + bar.y0)        
        ax.annotate(val, (bar.x0, bar.y1 + y_offset))
    # customize plot and save
    ax.set_ylabel("Fraction of null values")
    ax.set_ylim(0,0.3)
    plt.tight_layout()
    plt.savefig('plots/null_fraction.png')
    return

def plot_null_vs_time(pdf, time, columns, n_rows=4, n_columns=4, figsize=(20,12)):
    '''
    Plots fraction of null data as a function of time for each column.
    Args:
        pdf: pandas dataframe
        time: time column
        columns: columns of interest
    Return: None
    '''
    # calculate percecntage of valid data for each month
    pdf_null = pdf.groupby(time).apply(lambda x: x.isnull().mean())\
                  .sort_index()\
                  .drop([time], axis=1)
    # create figure and axes
    fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize)
    ax = ax.flatten()
    # count number of non-null data for each feature grouped by month
    columns = [x for x in columns if x != time] # remove time column
    for i, column in enumerate(columns):
        # fraction of null YOY changes
        ax[i].plot(pdf_null.index, pdf_null[column])
        # customize axes
        ax[i].set_xlabel(column)
        ax[i].set_ylabel("Missing data (%)")
        ax[i].set_ylim(0,1)
        #ax[i].set_xlim(x1, x2)
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(90)
    # remove extra subplots
    for x in np.arange(len(columns),len(ax),1):
        fig.delaxes(ax[x])
    plt.tight_layout()
    plt.savefig('plots/null_fraction_vs_time.png')
    plt.cla()
    return

      
def plot_correlation_matrix(df, columns, figsize=(5,5)):
    '''
    Plots correlation matrix of the given columns
    Args:
        pdf: Spark dataframe
        others: plotting options
    Return: None
    '''
    # drop null values to calculate correlation
    df_dropna = df.dropna("any")
    # initiate vector assembler
    assembler = VectorAssembler(inputCols=columns, outputCol='column_vector')
    # vectorize features
    df_vectorized = assembler.transform(df_dropna).select("column_vector")
    # calculate correlation between features
    correlation_mat = Correlation.corr(df_vectorized, "column_vector").collect()[0][0]
    # convert dense matrix to array
    correlation_array = correlation_mat.toArray()
    
    # create figure and axes
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # generate a mask for the upper triangle
    mask = np.zeros_like(correlation_array, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # make heatmap of correlation matrix
    sns.heatmap(correlation_array, xticklabels=columns, yticklabels = columns, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    # customize plot
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.cla()
    return

if __name__ == "__main__":

    #----------------------------------------------
    # load input data
    #----------------------------------------------
    
    # define features to be used
    features = ['CAP', 'AG', 'ROA', 'EG', 'LTG', 'SG', 'CVROIC', 'GS', 'SEV', 'FCFA']
    
    # create spark session
    spark = SparkSession.builder.appName("spark").getOrCreate()
    
    # import raw csv into spark dataframe
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # convert eom from int to timestamp
    df = df.withColumn("eom", udf(lambda x: str(x))("eom"))
    df = df.withColumn("eom", to_timestamp(df.eom, 'yyyyMM'))

    # create pdf for plotting
    pdf = df.toPandas()

    #----------------------------------------------
    # exploratory data Analysis (EDA)
    #----------------------------------------------
    
    # Feature distribution
    '''
    Before we preprocess data, individual variables are explored by examining their distributions.
    The goal of this section is to understand:
    - the validity of the preprocessing method (winsorizing, standardization)
    - any invalid outliers. Invalid outliers can be handled by winsorization.
    '''

    if perform_eda:
        # plot distribution of all columns
        plot_dist_features(df, df.columns, n_rows=5, n_columns=3, n_bins=50, figsize=(10,10), log=False)
        plot_dist_features(df, df.columns, n_rows=5, n_columns=3, n_bins=50, figsize=(10,10), log=True)

        # plot fraction of null values
        plot_null(df, df.columns, figsize=(15,5))

        # plot fraction of null values as a function of time
        plot_null_vs_time(pdf, time="eom", columns=df.columns, n_rows=4, n_columns=4, figsize=(20,20))
          
        # plot feature correlation
        plot_correlation_matrix(df=df, columns=features, figsize=(10,10))


#    
#    # COMMAND ----------
#    
#    #SecurityID
#    def plot_scatter(ax, pair, xlim=None, ylim=None):
#      
#      # get values of each column
#      values_x = np.array(df_dropna.select(pair[0]).collect())
#      values_y = np.array(df_dropna.select(pair[1]).collect())
#    
#      # draw scatter plot
#      ax.scatter(values_x, values_y)
#      
#      # customize axes
#      ax.set_xlabel(pair[0])
#      ax.set_ylabel(pair[1])
#      
#      if xlim:
#        ax.set_xlim(xlim)
#      if ylim:
#        ax.set_ylim(ylim)
#      
#      return ax
#    
#    # COMMAND ----------
#    
#    if perform_eda:
#      
#      # drop null values
#      df_dropna = df.dropna("any")
#      
#      # pairs to plot
#      pairs = [("SG", "CVROIC"), ("LTG", "GS")]
#      xlim = [None, (-100,100)]
#      
#      # create figure and axes
#      fig, ax = plt.subplots(1,2, figsize=(5,3))
#    
#      for i, a in enumerate(ax):
#        a = plot_scatter(a, pairs[i], xlim=xlim[i])
#    
#      # customize plot
#      plt.tight_layout()
#    
#      # display plot
#      display(fig)
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Define functions for preprocessing
#    
#    # COMMAND ----------
#    
#    # MAGIC %md Each variables are processed with the following procedures.
#    # MAGIC 1. Winsorization
#    # MAGIC 2. Standardization
#    # MAGIC 3. Imputation
#    
#    # COMMAND ----------
#    
#    ''' Step 1: Winsorization'''
#    
#    '''Winsorize each var by month
#    Note: use pandas quantitle function instead of scipy.winsorize function,
#    because of Scipy winsorize function has NaN handling bug, as noted in previous text'''
#    
#    def winsorize_series(s):
#        '''input: a Series
#           return: winsorized series'''
#        s = pd.Series(s)
#        q = s.quantile([winsorize_alpha_lower, winsorize_alpha_upper])
#        if isinstance(q, pd.Series) and len(q) == 2:
#            s[s < q.iloc[0]] = q.iloc[0]
#            s[s > q.iloc[1]] = q.iloc[1]    
#        return s
#    
#    # COMMAND ----------
#    
#    @pandas_udf(data_schema, functionType=PandasUDFType.GROUPED_MAP)
#    def winsorize_df(df):
#        '''input: a DataFrame
#        return apply winsorize function above on each column''' 
#        for feature in features:
#          if not feature == "GS": # do not perform winsorization on GS
#            df[feature] = winsorize_series(df[feature])
#        return df
#    
#    # COMMAND ----------
#    
#    def standardize_series(col):
#        '''Normalize each column by month mean and std'''
#        return (col - col.mean()) / col.std()
#    
#    # COMMAND ----------
#    
#    @pandas_udf(data_schema, functionType=PandasUDFType.GROUPED_MAP)
#    def standardize_df(df):
#        '''Normalize dataframe by month mean and std'''
#        for feature in features:
#          df[feature] = standardize_series(df[feature])    
#        return df
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Apply preprocessing
#    
#    # COMMAND ----------
#    
#    if run_preprocessing:
#      
#      # apply winsorization
#      df_winsorized = df.groupby("eom").apply(winsorize_df)
#    
#      # standardize features
#      df_standardized = df_winsorized.groupby("eom").apply(standardize_df)
#      
#    else:
#      # do not perform any transformation
#      df_winsorized = df_standardized = df
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Imputing missing data
#    
#    # COMMAND ----------
#    
#    # MAGIC %md The simple way to impute data is to replace missing values with mean or zero Z score. However, the EDA shows that some features have a large fraction (>10%) of missing values (CVROIC, GS, SEV). The EDA also suggests that there are some correlation between these features and other variables such as SecurityID. Therefore, in this section, two different approaches are compared for imputing data.
#    # MAGIC 
#    # MAGIC 1. Replace with the mean Z score within the same month which is zero.
#    # MAGIC 2. Replace with the mean Z score within the same Security group.
#    # MAGIC 3. Replace with the mean Z score within the same Security group.
#    # MAGIC 
#    # MAGIC In order to compare the performance of these two methods, a subset of data is masked with Null values and used as a test set. i.e. Known values are imputed, and the mean squared errors are compared between two methods.
#    
#    # COMMAND ----------
#    
#    def impute_by_month(df):
#        '''Impute missing data with the mean Z score within the same month group'''
#        return df.fillna(0, subset=features)
#    
#    # COMMAND ----------
#    
#    @pandas_udf(data_schema, functionType=PandasUDFType.GROUPED_MAP)
#    def impute_by_securityID(df):
#        '''Impute missing data with the mean Z score within the same security ID'''    
#        for feature in features:
#            mean = df[feature].mean()
#            df[feature] = df[feature].fillna(mean)
#            # if mean is not available, impute with zero Z score
#            df[feature] = df[feature].fillna(0)
#        return df
#    
#    # COMMAND ----------
#    
#    @pandas_udf(data_schema, functionType=PandasUDFType.GROUPED_MAP)
#    def impute_by_securityID_forward(df):
#        '''Impute missing data with the previous Z score of the same security ID'''    
#        for feature in features:
#            df[feature] = df.sort_values("eom")[feature].fillna(method='ffill')
#            # if mean is not available, impute with zero Z score
#            df[feature] = df[feature].fillna(0)
#        return df
#    
#    # COMMAND ----------
#    
#    # MAGIC %md Creating test dataset by masking 10% of known data.
#    
#    # COMMAND ----------
#    
#    if impute_test:
#      # split dataframe into two. Smaller subset will be replaced with null
#      df_split = df_standardized.dropna("any").randomSplit([0.1, 0.9])
#    
#      # dataframe with ground-truth values
#      df_impute_truth = df_split[0]
#    
#      # create null columns to impute
#      for feature in features:
#        df_split[0] = df_split[0].withColumn(feature, lit(None).cast(DoubleType()))
#    
#      # rename dataframe with null values
#      df_impute_null = df_split[0]
#    
#      # dataframe including 10% of null values to impute
#      df_impute_test = df_impute_null.union(df_split[1])
#    
#    # COMMAND ----------
#    
#    if impute_test:
#    
#      # impute missing data by mean Z score calculated within the same month
#      df_impute_month = impute_by_month(df_impute_test)
#      
#      # impute missing data by mean Z score calculated within the same securityID  
#      df_impute_securityID = df_impute_test.groupby("SecurityID").apply(impute_by_securityID)
#    
#      # impute missing data with the previous Z score of the same security ID
#      df_impute_securityID_forward = df_impute_test.groupby("SecurityID").apply(impute_by_securityID_forward)
#    
#    # COMMAND ----------
#    
#    if impute_test:
#      # to estimate performance, select only test set after the imputation. sql-like join
#      df_impute_month = df_impute_month.join(df_impute_null.select(["SecurityID", "eom"]), ["SecurityID", "eom"])
#      df_impute_securityID = df_impute_securityID.join(df_impute_null.select(["SecurityID", "eom"]), ["SecurityID", "eom"])
#      df_impute_securityID_forward = df_impute_securityID_forward.join(df_impute_null.select(["SecurityID", "eom"]), ["SecurityID", "eom"])
#    
#    # COMMAND ----------
#    
#    if impute_test:
#      # arrays to hold MSE
#      mse_impute_by_month = []
#      mse_impute_by_securityID = []
#      mse_impute_by_securityID_forward = []
#    
#      # calculate mean squared error
#      for feature in features:
#        # convert columns into numpy array to calculate mean squared error
#        array_impute_month = np.array(df_impute_month.select(feature).collect())
#        array_impute_securityID = np.array(df_impute_securityID.select(feature).collect())
#        array_impute_securityID_forward = np.array(df_impute_securityID_forward.select(feature).collect())
#        array_impute_truth = np.array(df_impute_truth.select(feature).collect())
#        
#        # calculate mse
#        mse_month = ((array_impute_truth - array_impute_month) ** 2).mean(axis=0)[0]
#        mse_securityID = ((array_impute_truth - array_impute_securityID) ** 2).mean(axis=0)[0]
#        mse_securityID_forward = ((array_impute_truth - array_impute_securityID_forward) ** 2).mean(axis=0)[0]
#    
#        # append to result
#        mse_impute_by_month.append(mse_month)
#        mse_impute_by_securityID.append(mse_securityID)
#        mse_impute_by_securityID_forward.append(mse_securityID_forward)    
#    
#    # COMMAND ----------
#    
#    if impute_test:
#      # compare mse of two methods
#      pdf_mse = pd.DataFrame([mse_impute_by_month,
#                              mse_impute_by_securityID,
#                              mse_impute_by_securityID_forward], columns=features)
#      display(spark.createDataFrame(pdf_mse))
#    
#    # COMMAND ----------
#    
#    if impute_test:
#      
#      # set number of columns
#      n_rows = 4
#      n_columns = 3
#      
#      # create figure and axes
#      fig, ax = plt.subplots(n_rows, n_columns, figsize=(12,8))
#      
#      for i, feature in enumerate(features):
#      
#        # get row and column index given integer
#        row = get_matrix_index(i, n_columns)[0]
#        column = get_matrix_index(i, n_columns)[1]  
#    
#        # convert columns into numpy array to calculate mean squared error
#        array_impute_month = np.array(df_impute_month.select(feature).collect())
#        array_impute_securityID = np.array(df_impute_securityID.select(feature).collect())
#        array_impute_securityID_forward = np.array(df_impute_securityID_forward.select(feature).collect())
#        array_impute_truth = np.array(df_impute_truth.select(feature).collect())  
#        
#        # set bin size
#        bins=np.histogram(np.hstack((array_impute_month,array_impute_securityID)), bins=30)[1] 
#    
#        # make histograms
#        ax[row][column].hist(array_impute_month - array_impute_truth, bins=bins)
#        ax[row][column].hist(array_impute_securityID - array_impute_truth, bins=bins, alpha=0.4)
#        ax[row][column].hist(array_impute_securityID_forward - array_impute_truth, bins=bins, alpha=0.4)
#        
#        # customize axes
#        ax[row][column].set_xlabel(feature)
#        ax[row][column].set_ylabel("Entries")
#        ax[row][column].set_xticks(ax[row][column].get_xticks()[::2])    
#    
#      display(fig)
#    
#    # COMMAND ----------
#    
#    # finally impute date
#    if impute_data:
#      df_preprocessed = impute_by_month(df_standardized)
#      #df_preprocessed = df_standardized.groupby("SecurityID").apply(impute_by_securityID_forward)
#      #df_preprocessed = df_standardized.groupby("SecurityID").apply(impute_by_securityID)
#    else:
#      df_preprocessed = df_standardized
#      
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Examining preprocessing result
#    
#    # COMMAND ----------
#    
#    if debug:
#      display(df_preprocessed.describe().select(["summary"]+features))
#    
#    # COMMAND ----------
#    
#    # temporary caching for development
#    if debug:
#      df.cache()
#      df_winsorized.cache()
#      df_standardized.cache()
#    
#    # COMMAND ----------
#    
#    # sample subset of data for a specific month
#    month = '201505'
#    
#    if check_processed_data:
#      # drop all rows containing null value for EDA purpose
#      df_dropna = df.dropna("any")
#      df_winsorized_dropna = df_winsorized.dropna("any")
#      df_standardized_dropna = df_standardized.dropna("any")
#    
#      # filter dataframe by specific month for plotting
#      df_dropna_month = df_dropna.filter(df_dropna.eom == month)
#      df_winsorized_dropna_month = df_winsorized_dropna.filter(df_winsorized_dropna.eom == month)
#      df_standardized_dropna_month = df_standardized_dropna.filter(df_standardized_dropna.eom == month)
#    
#    # COMMAND ----------
#    
#    # plot processed data
#    if check_processed_data:
#    
#      # set number of columns
#      n_rows = 6
#      n_columns = 5
#    
#      # set number of bins
#      n_bins = 25
#    
#      # create figure and axes
#      fig, ax = plt.subplots(n_rows, n_columns, figsize=(12,12))
#    
#      # columns to check
#      cols = ["AG", "EG", "LTG", "ROA", "SG", "CAP", "CVROIC", "SEV", "GS"]
#    
#      # loop over each columns
#      for i, feature in enumerate(cols):
#    
#          # get row and column index given integer
#          row = get_matrix_index(i, n_columns)[0] * 3
#          column = get_matrix_index(i, n_columns)[1]
#    
#          # get values of each column
#          values_raw = np.array(df_dropna_month.select(feature).collect())
#          values_winsorized = np.array(df_winsorized_dropna_month.select(feature).collect())
#          values_standardized = np.array(df_standardized_dropna_month.select(feature).collect())
#    
#          # set x range
#          x_range_raw = (values_winsorized.min(), values_winsorized.max())
#          x_range_winsorized = x_range_raw
#          x_range_standardized = (values_standardized.min(), float(values_standardized.max()))
#    
#          if False:
#            if feature == "CAP":
#              x_range_raw = (0, 10000)
#              x_range_winsorized = x_range_raw
#    
#            if feature == "CVROIC":
#              x_range_raw = (0, 0.5)
#              x_range_winsorized = x_range_raw
#    
#            if feature == "SEV":
#              x_range_raw = (0, 0.5)
#              x_range_winsorized = x_range_raw
#    
#          # draw histograms
#          ax[row][column].hist(values_raw, bins=n_bins, range=x_range_raw, color= 'r')
#          ax[row+1][column].hist(values_winsorized, bins=n_bins, range=x_range_winsorized, alpha = 0.7, color= 'b')
#          ax[row+2][column].hist(values_standardized, bins=n_bins, range=x_range_standardized, alpha = 0.7, color= 'g')
#    
#          # customize axes
#          ax[row][column].set_ylabel("Raw")
#          ax[row+1][column].set_ylabel("Winsorized")
#          ax[row+2][column].set_ylabel("Standardized")
#    
#          ax[row][column].set_xlabel(feature)
#          ax[row+1][column].set_xlabel(feature)
#          ax[row+2][column].set_xlabel(feature)
#    
#          ax[row][column].set_xticks(ax[row][column].get_xticks()[::2])
#          ax[row+1][column].set_xticks(ax[row+1][column].get_xticks()[::2])    
#          ax[row+2][column].set_xticks(ax[row+2][column].get_xticks()[::2])    
#    
#      # customize plot
#      plt.tight_layout()
#    
#      # display plot
#      display(fig)
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Delete all the observations where fmTotalReturn or fqTotalReturn are missing
#    
#    # COMMAND ----------
#    
#    # MAGIC %md Before saving the preprocessed data, the observations are removed if fmTotalReturn or fqTotalReturn is missing. The fraction of missing returns are less than 0.5%.
#    
#    # COMMAND ----------
#    
#    if run_preprocessing:
#      
#      # define target variables
#      targets = ["fmTotalReturn", "fqTotalReturn"]
#      
#      # count total number of rows and null values for each feature
#      n_total = np.array([df_preprocessed.count() for target in targets])
#      n_null_raw = np.array([df_preprocessed.where(col(target).isNull()).count() for target in targets])
#    
#      # calculate fraction of null values for each feaure
#      fraction_null_raw = n_null_raw / n_total
#    
#      # display fraction
#      print(pd.DataFrame([fraction_null_raw], columns=targets))
#    
#    # COMMAND ----------
#    
#    if run_preprocessing:
#      # drop missing values
#      df_preprocessed = df_preprocessed.dropna(subset=["fmTotalReturn", "fqTotalReturn"])  
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Keep only first 2 digits of GICSSubIndustryNumber
#    
#    # COMMAND ----------
#    
#    if run_preprocessing:
#      
#      # fill empty GICSSubIndustryNumber with 99999999
#      df_preprocessed = df_preprocessed.fillna({"GICSSubIndustryNumber":99999999})
#    
#      # keep only first two digits
#      df_preprocessed = df_preprocessed.withColumn("GICSSubIndustryNumber", udf(lambda x:str(x)[:2])(df_preprocessed.GICSSubIndustryNumber))
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Create classification label
#    
#    # COMMAND ----------
#    
#    # def create_classification_label(df, label):
#    #   '''
#    #   Create classification label given continuous label
#    #     class_binary: 1 if label is greater than 0. zero otherwise.
#    #     class_quintile: Assign label by quintile (5 classes).
#    #     class_tertiles: Assign label by tertiles (3 classes).
#        
#    #   Args:
#    #     df: spark dataframe containing label
#    #   Return:
#    #     df: spark dataframe with new classification labels
#    #   '''
#      
#    #   # create binary class label
#    #   df = df.withColumn(label+"_binary", udf(lambda x: 1 if x > 0 else 0, IntegerType())(col(label)))
#    
#    #   # create quintile label
#    #   qd_quintile = QuantileDiscretizer(numBuckets=5, inputCol=label, outputCol=label+"_quintile")
#    #   df = qd_quintile.fit(df).transform(df)
#      
#    #   # create tertiles label
#    #   qd_tertile = QuantileDiscretizer(numBuckets=3, inputCol=label, outputCol=label+"_tertile")
#    #   df = qd_tertile.fit(df).transform(df)
#      
#    #   return df
#    
#    # COMMAND ----------
#    
#    def create_classification_label(df, label):
#      '''
#      Assign classification label created within each month
#      Args:
#        df: spark dataframe
#        label: target label. ex. "fmTotalReturn"
#      '''  
#      # define schema with additional column 
#      schema = StructType([StructField(label+"_quintile", IntegerType(), True),
#                           StructField(label+"_decile", IntegerType(), True)]
#                          + df.schema.fields) 
#         
#      @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
#      def assign_label(pdf):
#        # assign quintile label
#        pdf[label+"_quintile"] = pd.qcut(x=pdf[label], q=5,
#                                         labels=["4", "3", "2", "1", "0"])
#        # assign decile label
#        pdf[label+"_decile"] = pd.qcut(x=pdf[label], q=10,
#                               labels=["9", "8", "7", "6", "5", "4", "3", "2", "1", "0"])
#        
#        # convert category type to string
#        pdf[label+"_quintile"] = pdf[label+"_quintile"].astype(int)
#        pdf[label+"_decile"] = pdf[label+"_decile"].astype(int)
#        
#        return pdf
#    
#      return df.groupby('eom').apply(assign_label)
#    
#    # COMMAND ----------
#    
#    if create_label:
#      labels = ["fmTotalReturn", "fqTotalReturn"]
#    
#      # create class labels
#      for label in labels:
#        df_preprocessed = create_classification_label(df_preprocessed, label)
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Feature engineering
#    
#    # COMMAND ----------
#    
#    def add_second_order_terms(df, variables):
#      ''' add second order terms to dataframe given numerical features
#      Args:
#        df: A dataframe containing numerical features
#        variables: Names of features
#      Returns:
#        df: A dataframe with additioanl features, x2 = x*x 
#        second_order_term: list of second order terms
#      '''
#      # add second order terms
#      second_order_terms = [var+"2" for var in variables]
#      for var in variables:
#        df = df.withColumn(var+"2", col(var) * col(var))    
#      return df, second_order_terms
#    
#    # COMMAND ----------
#    
#    def add_interaction_terms(df, variables):
#      ''' add interaction terms to dataframe given numerical features
#      Args:
#        df: A dataframe containing numerical features
#        variables: Names of features
#      Returns:
#        A dataframe with additioanl features, xy = x*y
#        interaction_terms: list of interaction terms
#      '''
#      # add interaction terms
#      interaction_terms = [(x, y) for i, x in enumerate(variables) 
#                                  for j, y in enumerate(variables) if j < i]
#      for interaction in interaction_terms:
#        var1 = interaction[0]
#        var2 = interaction[1]
#        df = df.withColumn(var1+"-"+var2, col(var1) * col(var2))
#      return df, interaction_terms
#    
#    # COMMAND ----------
#    
#    def add_quotient_terms(df, variables):
#    # add quotient terms
#      def calculate_quotient(cols):
#        if cols[1] != 0.0:
#          col = cols[0] / cols[1]
#        else:
#          col = None
#        return col
#    
#      # add quotien terms
#      quotient_terms = [(x, y) for i, x in enumerate(variables) 
#                                  for j, y in enumerate(variables) if x != y]
#      
#      for quotient in quotient_terms:
#        var1 = quotient[0]
#        var2 = quotient[1]
#        quot = var1+"/"+var2
#        df = df.withColumn(quot, udf(calculate_quotient, DoubleType())(array(var1, var2)))
#        
#        # replace Null quotient with max value of the same column
#        max_quot = df.select(col_max(col(quot)).alias("max")).collect()[0][0]
#        df = df.withColumn(quot, udf(lambda x: max_quot if x is None else x, DoubleType())(col(quot)))
#      return df, quotient_terms
#    
#    # COMMAND ----------
#    
#    if feature_engineering:
#      # add additional features
#      df_preprocessed, second_order_terms = add_second_order_terms(df_preprocessed, features)
#      df_preprocessed, interaction_terms = add_interaction_terms(df_preprocessed, features)
#      df_preprocessed, quotient_terms = add_quotient_terms(df_preprocessed, features)
#    
#    # COMMAND ----------
#    
#    # MAGIC %md ## Save preprocessed data
#    
#    # COMMAND ----------
#    
#    if run_preprocessing:
#    #if False:  
#      # save as csv
#      #df_preprocessed.write.csv('/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context_r2p3.2digit_GICSSubIndustryNumber.csv', header=True, mode="overwrite")
#      #df_preprocessed.write.csv('/mnt/mainblob/asset_growth/data/Data_for_AssetGrowth_Context_r2p4.ImputeBySecurityID.csv', header=True, mode="overwrite")
#      df_preprocessed.write.csv('/mnt/mainblob/asset_growth/data/AG_r2p6.classLabelByMonth.csv', header=True, mode="overwrite")
#    
#    # COMMAND ----------
#    
#    df_preprocessed.printSchema()
#    
#    # COMMAND ----------
#    
#    display(df_preprocessed)
#    
#    # COMMAND ----------
#    
#    # check if there is any missing value
#    n_null={}
#    for column in df_preprocessed.columns:
#      n_null[column] = df_preprocessed.filter(col(column).isNull()).count()
#    
#    # COMMAND ----------
#    
#    n_null
