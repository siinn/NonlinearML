#!/usr/bin/env python
## Import custom libraries
import pandas as pd
import numpy as np
import NonlinearML.lib.utils as utils
import NonlinearML.plot.backtest as plot_backtest

#-------------------------------------------------------------------------------
# Set configuration
#-------------------------------------------------------------------------------
# Set features of interest
""" Available features:
'GICSSubIndustryNumber', 'CAP', 'AG', 'ROA', 'ES', 'LTG', 'SG', 'CVROIC',
'GS', 'SEV', 'FCFA', 'ROIC', 'Momentum' """
feature_x = 'AG'
feature_y = 'FCFA'

# Set path to save output figures
output_path = 'output/%s_%s/' % (feature_x, feature_y)

# Set labels
n_classes=3
class_label={0:'T1', 1:'T2', 2:'T3'}
suffix="descrite"

# Set output label classes
label_reg = "fqTotalReturn" # continuous target label
label_cla = "_".join([label_reg, suffix]) # discretized target label
label_fm = "fmTotalReturn" # monthly return used for calculating cum. return

# Set data column
date_column = "eom"

# Set metric for training
cv_metric = 'f1-score'

# Set list of models to include
models = ['lr', 'nn', 'knn']


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":


    # Read results from all models
    cum_return_test = {}
    cv_results = {}

    for model in models:
        print("Reading results from model: %s" % model)
        path = "/".join([output_path.strip("/"), model, "csv/"])
        # Read csv
        cum_return_test[model] = pd.read_csv(
            path+"cum_return_test.csv",
            parse_dates=[date_column], infer_datetime_format=True)
        cv_results[model] = pd.read_csv(path+"cv_results.csv")

    # Plot comparison of cumulative return
    print("Plotting the comparison of cumulative returns..")
    plot_backtest.plot_cumulative_return_diff(
        list_cum_returns=list(cum_return_test.values()),
        list_labels=list(cum_return_test.keys()),
        label_reg=label_reg,
        date_column=date_column,
        figsize=(8, 6), return_label=sorted(np.arange(n_classes)),
        kwargs_train={'ylim':(-1, 3)},
        kwargs_test={'ylim':(-1, 3)},
        legend_order=['lr', 'nn', 'knn'],
        filename=output_path+"model_comparison/return_diff_summary")

    
    # Save summary
    print("Saving summary of model comparison..")
    utils.save_summary(
        class_label=class_label, metric=cv_metric,
        output_path=output_path+"model_comparison/",
        cv_results=cv_results)


    print("Successfully completed all tasks")


