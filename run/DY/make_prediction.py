#!/usr/bin/env python
# Import common python libraries
import matplotlib
#import numpy as np
#import os
import pandas as pd
import pickle
import sys
import warnings

# Import custom libraries
import NonlinearML.lib.io as io
import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db
from xgboost.sklearn import XGBRegressor


# Supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None 

#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/Production.20200305.p1.csv'

# Set features of interest
features = ['DY_dmed', 'PO_dmed']

# Set number of bins for ranking
rank_n_bins=10
rank_label={x:'D'+str(x) for x in range(rank_n_bins)}
rank_order=[9,8,7,6,5,4,3,2,1,0] # High return to low return
rank_top = 9
rank_bottom = 0

# Set path to save output figures
model_tag = 'v1'
output_path = 'output/DY/Production/%s/' % "_".join(features)
MODEL_PATH = '/mnt/mainblob/nonlinearML/NonlinearML/model/DY/DY_dmed_PO_dmed/%s/enhanced_DY_xgboost.%s' %(model_tag, model_tag)

# Set data column
date_column = "smDate"

# Set security ID column
security_id = 'SecurityID'


# Set color scheme for decision boundary plot
cmap = matplotlib.cm.get_cmap('RdYlGn')
db_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

cmap_scatter = matplotlib.cm.get_cmap('RdYlGn', rank_n_bins)
db_colors_scatter = [matplotlib.colors.rgb2hex(cmap_scatter(i)) 
    for i in range(cmap_scatter.N)]

# Set decision boundary plotting options
db_xlim = (-1.5, 3)
db_ylim = (-3, 3)
db_res = 0.01
db_figsize= (10, 8)
db_annot_x=0.02
db_annot_y=0.98
db_nbins=50
db_vmin=-0.30
db_vmax=0.30



#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set logging configuration
    io.setConfig(path=output_path, filename="log.txt")

    # Read input csv
    io.title("Reading input file")
    io.message("\t> Input = %s" % INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, index_col=None)

    # Load model
    io.title("Loading trained model")
    io.message("\t> Model path = %s" % MODEL_PATH)
    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    # Print model parameters
    if 'get_params' in dir(model):
        io.message("Model parameters:")
        io.message(["=".join([key.strip('\''),str(model.get_params()[key])])\
            for key in model.get_params()])
    io.message("Features: %s" %features)

    # Make prediction
    io.title("Making predictions")
    df['pred'] = model.predict(df[features], validate_features=False)


    #---------------------------------------------------------------------------
    # Prediction distribution
    #---------------------------------------------------------------------------
    # Plot distribution of prediction
    io.message("Plotting prediction distribution")
    df_MSCIEM = df.loc[df['IsMSCIEM']==1]
    df_nonMSCIEM = df.loc[df['IsMSCIEM']==0]

    # All data
    plot.plot_distribution(
        df=df, columns=['pred'], n_rows=1, n_columns=1,
        bins=[50], ylog=[False], title=[""],
        x_label=['Prediction'], y_label=['Samples'],
        figsize=(8,6), filename=output_path+"dist_pred",
        show_sigma=False, vlines=None)

    # MSCIEM==1
    plot.plot_distribution(
        df=df_MSCIEM, columns=['pred'], n_rows=1, n_columns=1,
        bins=[50], ylog=[False], title=[""],
        x_label=['Prediction'], y_label=['Samples'],
        figsize=(8,6), filename=output_path+"dist_pred_MSCIEM",
        show_sigma=False, vlines=None)

    # MSCIEM==0
    plot.plot_distribution(
        df=df_nonMSCIEM, columns=['pred'], n_rows=1, n_columns=1,
        bins=[50], ylog=[False], title=[""],
        x_label=['Prediction'], y_label=['Samples'],
        figsize=(8,6), filename=output_path+"dist_pred_nonMSCIEM",
        show_sigma=False, vlines=None)

    #---------------------------------------------------------------------------
    # Decision boundary
    #---------------------------------------------------------------------------
    # Plot decision boundary of the best model with scattered plot.
    io.message("Plotting decision boundaries")
    plot_db.decision_boundary(
        model=model, df=df, features=features, h=db_res,
        x_label=features[0], y_label=features[1],
        colors=db_colors,
        xlim=db_xlim, ylim=db_ylim,
        vmin=db_vmin, vmax=db_vmax,
        figsize=db_figsize,
        colorbar=True, ticks=None,
        scatter=True, subsample=0.01, label_cla=None,
        scatter_legend=False, colors_scatter=db_colors_scatter,
        dist=True, nbins=db_nbins,
        filename=output_path+"decision_boundary")

    #---------------------------------------------------------------------------
    # Save predictions as csv
    #---------------------------------------------------------------------------
    df.to_csv(output_path+'pred.csv')






