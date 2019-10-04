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
        # Get id of selected model
        id_selected_model = id_selected[model]['id_selected_model'][0]
        best_model = cv_results[model].drop(
            cv_results[model].filter(regex='value').columns, axis=1)\
            .loc[id_selected_model]
        best_model['Model'] = model
        # Append to summary results
        df_summary = df_summary.append(best_model)
    df_summary = df_summary.sort_index(axis=1)
    # Save the summary
    utils.create_folder(output_path+'summary.csv')
    df_summary.to_csv(output_path+'summary.csv')
    return
        




def model_comparison(
    models, output_path, label_reg, class_label, col_pred,
    date_column, ylim=(-1,5), figsize=(8,6)):
    """ Compare cumulative return of all models. This function reads
    models results from csv.
    Args:
        models: List of models. Ex. ['lr', 'xgb']
        output_path: Parent path to saved model results
        label_reg: Regression label. Ex. 'fmTotalReturn'.
            Used to label y axis
        class_label: List of return labels in order of [high, medium, low]
        date_column: Ex. 'eom' or 'smDate'
        others: Plotting options
    Return:
        csvs loaded into dataframe
    """
    # Initialize logger
    io.setConfig(path=output_path+"model_comparison/", filename="log")
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
        label_reg=label_reg, col_pred=col_pred,
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
    return {
        'cv_results': cv_results,
        'id_selcted': id_selected,
        'cum_return_test': cum_return_test}





def save_prediction(
    models, feature_x, feature_y, df_input, output_path, date_column):
    """ Add predictions to origianl input data and save as a new csv.
    This function reads predictions from csv.
    Args:
        models: List of models. Ex. ['lr', 'xgb']
        feature_x, feature_y: Features used in making predictions.
            Used to label prediction columns.
        df_input: Input dataframe
        output_path: Parent path to saved model results
        date_column: Ex. 'eom' or 'smDate'
    Return:
        csvs loaded into dataframe
    """
    # Initialize logger
    io.setConfig(path=output_path+"model_comparison/", filename="log")
    io.title("Save prediction") 

    # Read results from all models
    pred_test = {}
    pred_train = {}

    # Read csv
    for model in models:
        io.message("Reading predictions from model: %s" % model)
        try:
            path = "/".join([output_path.strip("/"), model, "csv/"])
            io.message(" > %spred_test.csv" % path)
            pred_test[model] = pd.read_csv(
                path+"pred_test.csv",
                parse_dates=[date_column], infer_datetime_format=True)
            io.message(" > %spred_train.csv" % path)
            pred_train[model] = pd.read_csv(
                path+"pred_train.csv",
                parse_dates=[date_column], infer_datetime_format=True)
        except:
            io.error("Cannot find model prediction.")
        
    # Reset index. Assumes that order isn't changed.
    """ NEED TO UPDATE THIS SO THAT WE CAN USE SECURITY ID TO JOIN"""
    df_output = df_input.reset_index()

    # Append model predictions to output dataframe
    for model in models:
        # Concatenate train and test prediction
        df_pred = pd.concat(
            [pred_train[model], pred_test[model]])
        # Append prediction to output
        df_output["_".join(["Pred", feature_x, feature_y, model])] = \
            df_pred.reset_index()['pred']

    # Save prediction
    io.message("Saving predictions as")
    output_file = output_path+"model_comparison/prediction.csv"
    io.message(" > %s" % output_file)
    utils.create_folder(output_file)
    df_output.to_csv(output_file)

    return {
        'pred_train': pred_train,
        'pred_test': pred_test,
        'df_output': df_output}



