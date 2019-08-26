import pandas as pd

import NonlinearML.lib.io as io
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.lib.utils as utils



def save_summary(output_path, cv_results, id_selected):
    '''Collect results from all models trained with best parameters and
        save them as csv
    Args:
        output_path: Path to save results
        cv_results: Dictionary of CV results obtained from grid search.
            Example: {'XGB': cv_results_xgb}
                where cv_results_xgb is output of grid_search_purged_cv
        id_selected: Dictionary containing id of the selected model
    Return:
        None
    '''
    df_summary = pd.DataFrame()
    for model in cv_results:
        # Append to summary results
        df_summary = df_summary.append(cv_results[model].loc[
            id_selected[model]['id_selected_model'][0],
            cv_results[model].columns.str.contains(
                'accuracy|precision|recall|f1-score')])
    # Save the summary
    utils.create_folder(output_path+'summary.csv')
    df_summary.to_csv(output_path+'summary.csv')
    return
        




def model_comparison(
    models, output_path, label_reg, class_label,
    date_column, ylim=(-1,5), figsize=(8,6)):
    """ Compare cumulative return of all models. This function reads
    models results from csv.
    Args:
        models: List of models. Ex. ['lr', 'xgb']
        output_path: Parent path to saved model results
        label_reg: Regression label. Ex. 'fqTotalReturn'
        class_label: List of class labels. Ex. [0, 1, 2, ..]
        date_column: Ex. 'eom' or 'smDate'
        others: Plotting options
    """
    io.title("Model comparison") 
    # Read results from all models
    cum_return_test = {}
    cv_results = {}
    id_selected = {}

    for model in models:
        io.message("Reading results from model: %s" % model)
        try:
            path = "/".join([output_path.strip("/"), model, "csv/"])
            # Read csv
            cum_return_test[model] = pd.read_csv(
                path+"cum_return_test.csv",
                parse_dates=[date_column], infer_datetime_format=True)
            cv_results[model] = pd.read_csv(path+"cv_results.csv")
            id_selected[model] = pd.read_csv(path+"id_selected_model.csv")
        except:
            io.error("Cannot find model results.")

    # Plot comparison of cumulative return
    io.message("Plotting the comparison of cumulative returns..")
    plot_backtest.plot_cumulative_return_diff(
        list_cum_returns=list(cum_return_test.values()),
        list_labels=list(cum_return_test.keys()),
        label_reg=label_reg,
        date_column=date_column,
        figsize=figsize, ylim=ylim,
        return_label=class_label,
        legend_order=models,
        filename=output_path+"model_comparison/return_diff_summary")
    # Save summary
    io.message("Saving summary of model comparison..")
    save_summary(
        output_path=output_path+"model_comparison/",
        cv_results=cv_results,
        id_selected=id_selected)
    return






