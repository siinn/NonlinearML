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
debug = True

# select algorithm to run    
perform_eda             = False
run_preprocessing       = True
impute_test             = False
impute_data             = True
create_label            = True
check_processed_data    = True

# set imputation method. available options: month, securityId_ff, securityId_average
impute_method           =  "month"

# set winsorization alpha
winsorize_alpha_lower = 0.05
winsorize_alpha_upper = 0.95    


# set plot style
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#plt.style.use('seaborn-talk')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'


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


    
def winsorize_series(s):
    '''Winsorize each var by month
    Note: use pandas quantitle function instead of scipy.winsorize function,
    because of Scipy winsorize function has NaN handling bug, as noted in previous text
    Args: a Series
    Return: winsorized series
    '''
    s = pd.Series(s)
    q = s.quantile([winsorize_alpha_lower, winsorize_alpha_upper])
    if isinstance(q, pd.Series) and len(q) == 2:
        s[s < q.iloc[0]] = q.iloc[0]
        s[s > q.iloc[1]] = q.iloc[1]    
    return s


def winsorize_df(df, features):
    '''input: a DataFrame
    return apply winsorize function above on each column''' 
    @pandas_udf(df.schema, functionType=PandasUDFType.GROUPED_MAP)
    def udf(pdf):
        for feature in features:
            pdf[feature] = winsorize_series(pdf[feature])
        return pdf
    return df.groupby("eom").apply(udf)


def standardize_series(col):
    '''Normalize each column by month mean and std'''
    return (col - col.mean()) / col.std()

def standardize_df(df, features):
    '''Normalize dataframe by month mean and std'''
    @pandas_udf(df.schema, functionType=PandasUDFType.GROUPED_MAP)
    def udf(df):
        for feature in features:
          df[feature] = standardize_series(df[feature])    
        return df
    return df.groupby("eom").apply(udf)

#----------------------------------------------
# define imputation methods
#----------------------------------------------

def impute_by_month(df):
    '''Impute missing data with the mean Z score within the same month group'''
    return df.fillna(0, subset=features)


def impute_by_securityID(df):
    '''Impute missing data with the mean Z score within the same security ID'''    
    @pandas_udf(df.schema, functionType=PandasUDFType.GROUPED_MAP)
    def udf(df):
        for feature in features:
            mean = df[feature].mean()
            df[feature] = df[feature].fillna(mean)
            # if mean is not available, impute with zero Z score
            df[feature] = df[feature].fillna(0)
        return df
    return df_standardized.groupby("SecurityID").apply(udf)

def impute_by_securityID_forward(df):
    '''Impute missing data with the previous Z score of the same security ID'''    
    @pandas_udf(df.schema, functionType=PandasUDFType.GROUPED_MAP)
    def udf(df):
        for feature in features:
            df[feature] = df.sort_values("eom")[feature].fillna(method='ffill')
            # if mean is not available, impute with zero Z score
            df[feature] = df[feature].fillna(0)
        return df
    return df_standardized.groupby("SecurityID").apply(udf)


#----------------------------------------------
# define plotting functions
#----------------------------------------------

def get_matrix_index(x, n):
        '''
        convert integer to the corresponding index in (m x n) matrix
        '''
        row = int(x/n)         # calculate row index
        column = x - row*n     # calculate column index
        return [row, column]

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

def plot_preprocessing_result(df_list, df_labels, columns, n_rows=6, n_columns=5, n_bins=25, figsize=(12,12)):
	'''
	Examine the preprocessing results. This function plots the distribution of given columns at each stage of preprocessing. 
	Args:
		df_list: list of three dataframes representing each step of preprocessing.
		df_label: label for each step.
		columns: columns of interest, typically features.
		others: plotting options.
	Return: None
	'''
    
    # create figure and axes
    fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize)

	# cache dataframes
	for df in df_list:
		df.cache()
		df.count()
    
    # loop over each columns
    for i, feature in enumerate(columns):
    
        # get row and column index given integer
        row = get_matrix_index(i, n_columns)[0] * 3
        column = get_matrix_index(i, n_columns)[1]
    
        # get values of each column
        values_raw = np.array(df_list[0].select(feature).collect())
        values_winsorized = np.array(df_list[1].select(feature).collect())
        values_standardized = np.array(df_list[2].select(feature).collect())
    
        # set x range
        x_range_raw = (values_winsorized.min(), values_winsorized.max())
        x_range_winsorized = x_range_raw
        x_range_standardized = (values_standardized.min(), float(values_standardized.max()))
    
        # draw histograms
        ax[row][column].hist(values_raw, bins=n_bins, range=x_range_raw, color= 'r', edgecolor='black')
        ax[row+1][column].hist(values_winsorized, bins=n_bins, range=x_range_winsorized, alpha = 0.7, color= 'b', edgecolor='black')
        ax[row+2][column].hist(values_standardized, bins=n_bins, range=x_range_standardized, alpha = 0.7, color= 'g', edgecolor='black')
    
        # customize axes
        ax[row][column].set_ylabel(df_labels[0])
        ax[row+1][column].set_ylabel(df_labels[1])
        ax[row+2][column].set_ylabel(df_labels[2])
    
        ax[row][column].set_xlabel(feature)
        ax[row+1][column].set_xlabel(feature)
        ax[row+2][column].set_xlabel(feature)
    
        ax[row][column].set_xticks(ax[row][column].get_xticks()[::2])
        ax[row+1][column].set_xticks(ax[row+1][column].get_xticks()[::2])    
        ax[row+2][column].set_xticks(ax[row+2][column].get_xticks()[::2])    
    
    # customize plot
    plt.tight_layout()
    plt.savefig('plots/proprocess_result.png')

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
    # Exploratory data Analysis (EDA)
    #----------------------------------------------
    '''
    Before we preprocess data, individual variables are explored by examining their distributions.
    The goal of this section is to understand:
    - the validity of the preprocessing method (winsorizing, standardization)
    - any invalid outliers. Invalid outliers can be handled by winsorization.
    '''

    if perform_eda:
		print("Creating EDA plots")
        # plot distribution of all columns
        plot_dist_features(df, df.columns, n_rows=5, n_columns=3, n_bins=50, figsize=(10,10), log=False)
        plot_dist_features(df, df.columns, n_rows=5, n_columns=3, n_bins=50, figsize=(10,10), log=True)

        # plot fraction of null values
        plot_null(df, df.columns, figsize=(15,5))

        # plot fraction of null values as a function of time
        plot_null_vs_time(pdf, time="eom", columns=df.columns, n_rows=4, n_columns=4, figsize=(20,20))
          
        # plot feature correlation
        plot_correlation_matrix(df=df, columns=features, figsize=(10,10))

    #----------------------------------------------
    # Run preprocessing
    #----------------------------------------------
    if run_preprocessing:
		print("Running preprocessing: winsorization and standardization")
        # apply winsorization. Not applied for AG
        df_winsorized = winsorize_df(df=df, features=[x for x in features if x != "GS"])
        # standardize features
        df_standardized = standardize_df(df=df_winsorized, features=[x for x in features if x != "GS"])

    else:
        # skip preprocessing
        df_standardized = df

    # cache processed data
    df_standardized.cache()
    df_standardized.count()
    
    #----------------------------------------------
    # Imputing missing data
    #----------------------------------------------
    # impute missing data
    if impute_data:
    	print("Imputing missing data using the method: %s" %impute_method)
        if impute_method == "month":
            df_preprocessed = impute_by_month(df_standardized)
        else if impute_method == "securityId_ff":
            df_preprocessed = impute_by_securityID(df_standardized)
        else if impute_method == "securityId_average":
            df_preprocessed = impute_by_securityID_forward(df_standardized)
        else:
            print("Impute method is not valid.")

    #----------------------------------------------
    # Examining preprocessing results
    #----------------------------------------------
    
    if check_processed_data:
		print("Examining preprocessing results")

        # sample subset of data for a specific month
        month = df.select('eom').collect()[0][0]        

        # drop all rows containing null value for EDA purpose
        df_dropna = df.dropna("any")
        df_winsorized_dropna = df_winsorized.dropna("any")
        df_standardized_dropna = df_standardized.dropna("any")
        
        # filter dataframe by specific month for plotting
        df_dropna_month = df_dropna.filter(df_dropna.eom == month)
        df_winsorized_dropna_month = df_winsorized_dropna.filter(df_winsorized_dropna.eom == month)
        df_standardized_dropna_month = df_standardized_dropna.filter(df_standardized_dropna.eom == month)

        # plot result
        plot_preprocessing_result(df_list=[df_dropna_month,
                                           df_winsorized_dropna_month,
                                           df_standardized_dropna_month],
                                  df_labels=["Raw", "Winsorized", "Standardized"],
                                  columns=features, n_rows=6, n_columns=5, n_bins=25, figsize=(20,20))
   
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
