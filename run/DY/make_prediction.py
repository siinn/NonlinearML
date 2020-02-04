#!/usr/bin/env python
# Import common python libraries
#from datetime import datetime
#import dateutil.relativedelta
import matplotlib
#import numpy as np
#import os
import pandas as pd
import pickle
import sys
import warnings

# Import custom libraries
#import NonlinearML.lib.stats as stats
#import NonlinearML.lib.summary as summary
import NonlinearML.lib.utils as utils
import NonlinearML.lib.io as io
import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db


# Supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('once')  # 'error', 'always', 'ignore'
pd.options.mode.chained_assignment = None 

#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set input and output path
INPUT_PATH = '/mnt/mainblob/nonlinearML/EnhancedDividend/data/production/DYPO_2020.01.30.p1.csv'

# Set features of interest
#features = ['DividendYield', 'Payout_E']
features = ['DY_dmed', 'PO_dmed']
#features = ['DividendYield', 'EG']
#features = ['DY_dmed', 'EG_dmed']

# Set number of bins for ranking
rank_n_bins=10
rank_label={x:'D'+str(x) for x in range(rank_n_bins)}
rank_order=[9,8,7,6,5,4,3,2,1,0] # High return to low return
rank_top = 9
rank_bottom = 0

# Set output label classes
#label_reg = 'fqRet' # continuous target label
#label_cla = 'QntfqRet' # discretized target label
#label_cla = 'fqRet_discrete' # discretized target label
#label_fm = 'fmRet' # monthly return used for calculating cum. return

# Set path to save output figures
output_path = 'output/DY/Production/%s/' % "_".join(features)
MODEL_PATH = 'model/DY/DY_dmed_PO_dmed/v1/xgb.pickle'

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
db_xlim = (-3, 3)
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
    df = pd.read_csv(INPUT_PATH, index_col=None, parse_dates=[date_column])

    # Load model
    io.title("Loading trained model")
    io.message("\t> Model path = %s" % MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Print model parameters
    if 'get_params' in dir(model):
        io.message("Model parameters:")
        io.message(["=".join([key.strip('\''),str(model.get_params()[key])])\
            for key in model.get_params()])
    io.message("Features: %s" %features)

    # Check if feature names matches
    if not features==model.get_booster().feature_names:
        io.message("Features don't match. Terminating job..")
        sys.exit("Features don't match. Terminating job..")

    # Make prediction
    io.title("Making predictions")
    df['pred'] = model.predict(df[features])

    #---------------------------------------------------------------------------
    # Examining output
    #---------------------------------------------------------------------------
    # Plot distribution of prediction
    io.message("Plotting prediction distribution")
    plot.plot_distribution(
        df=df, columns=['pred'], n_rows=1, n_columns=1,
        bins=[50], ylog=[False], title=[""],
        x_label=['Prediction'], y_label=['Samples'],
        figsize=(8,6), filename=output_path+"dist_pred",
        show_sigma=False, vlines=None)

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
        #annot={
        #    'text':utils.get_param_string(model.get_params()).strip('{}')\
        #        .replace('\'','').replace(',','\n').replace('\n ', '\n'),
        #    'x':db_annot_x, 'y':db_annot_y},
        scatter=True, subsample=0.01, label_cla=None,
        scatter_legend=False, colors_scatter=db_colors_scatter,
        dist=True, nbins=db_nbins,
        filename=output_path+"decision_boundary")



