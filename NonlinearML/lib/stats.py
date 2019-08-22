import itertools
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison

import NonlinearML.lib.io as io

def anova_test(df_data, verbose=False):
    """ Perform Anova test.

    Given K groups with n samples from each group, perform f-test and
    return f-stat and the corresponding p-values
    Args:
        data: Pandas dataframe representing data in the following format 
            model_i run_j, ..., run_n
            model_K run_j, ..., run_n
    Returns:
        f_stat, p_values
    """
    # Run ANOVA
    f_stat, p_value = stats.f_oneway(
        *[df_data.iloc[i].values for i in range(df_data.shape[0])])
    # Calculate degrees of freedom
    dof1 = df_data.shape[0] - 1 # K-1 where K is number of models
    dof2 = df_data.shape[1] - df_data.shape[0] # K-N
    # Print results
    if verbose:
        print(" >> F-statistics = %s, p-value = %s" % (f_stat, p_value))
        print(" >> DOF1 (K-1) = %s, DOF2 (N-K) = %s" % (dof1, dof2))
    return round(f_stat, 6), round(p_value, 6)

def post_hoc_test(df_data):
    """ Perform post hoc test (Tukey)
    Args:
        data: Pandas dataframe representing data in the following format
            model_i run_j, ..., run_n
            model_K run_j, ..., run_n
    Returns:
        df_posthoc: results in Pandas dataframe format
    """
    # Stack data from all groups
    df_stacked = df_data.stack().reset_index().rename(
        {'level_0': 'model','level_1': 'run',0: 'perf'}, axis=1)
    # Perform post hoc test
    post_hoc = MultiComparison(df_stacked['perf'], df_stacked['model'])
    df_posthoc = pd.DataFrame(post_hoc.tukeyhsd(alpha=0.05).summary().data)
    return df_posthoc

def select_best_model_by_anova(cv_results, cv_metric, param_grid, p_thres):
    """ Select best model by performing ANOVA and post-hoc test.
    
    The preferred model is selected among the models within the statistical
    uncertainties of the model with the highest score.
    
    Minimum index is selected because cv_results is sorted by the
    preferred order of hyperparameters and values in param_grid.

    Args:
        cv_results: Output of CV method (grid_search_purged_cv)
        cv_metric: Metric to use in model selction. e.g. 'f1-score'
        param_grid: Hyperparameter grid used to return best hyperparameters
        p_thres: p-value threshold for ANOVA. ex. 0.05
    Returns:
        f_stats: f-statistics of all models calculated using each metric
        p_values: p-values of all models calculated using each metric
        post_hoc_results: results of post-hoc tests 
        best_params: best parameter selected

    """
    def _convert_to_float(x):
        """ Attempts to convert to float if possible."""
        try:
            return float(x)
        except ValueError:
            return x

    # Extract list of metrics
    metric_names = [
        metric[:-7] for metric in cv_results.columns if 'values' in metric]
    metric_values = [
        metric for metric in cv_results.columns if 'values' in metric]

    # Convert strings to list in cv_results
    """ Example: '[0.0, 0.1, 0.5]' to ['0.0', '0.1', '0.5']. """
    if type(cv_results[metric_values].iloc[0,0]) == str:
        cv_results[metric_values] = cv_results[metric_values].applymap(
            lambda x: x.strip('[]').split(', '))

    # Perform ANOVA and post hoc test on each metrics
    f_stats = {}
    p_values = {}
    post_hoc_results = {}
    post_hoc_top = {}

    # Loop over each metric
    io.title("ANOVA and post hoc test on model performance")
    print("Performing ANOVA test..")
    for metric in metric_names:
        metric_values = metric+"_values"

        # Extract values of CV results
        model_perf = pd.DataFrame(cv_results[metric_values].tolist())\
                        .astype(float)
        # Perform one way ANOVA
        f_stats[metric], p_values[metric] = anova_test(model_perf)
        print("\t> %s: p-value = %s" % (metric, p_values[metric]))
        # If p-value is less than the threshold, perform post hoc test.
        if p_values[metric] < p_thres:
            post_hoc_results[metric] = post_hoc_test(model_perf)

    # Index of the model with the highest score
    id_max = cv_results[cv_metric].idxmax()
    print("\t> Model %s has the highest score" % id_max)

    # Check if we passed ANOVA given metric
    if cv_metric in post_hoc_results.keys():
        print("Performing post hoc test..")

        # Filter result by the model with the highest score
        post_hoc_top = post_hoc_results[cv_metric].loc[
            (post_hoc_results[cv_metric][0]==id_max) |
            (post_hoc_results[cv_metric][1]==id_max)]

        # Check if there are multiple models within statistical uncertainties
        num_candidates = post_hoc_top.loc[post_hoc_top[5]==False].shape[0]
        if num_candidates <= 1:
            print("\t > There is only one model with highest %s score." % cv_metric)
            id_selected_model = id_max
        else:
            print("\t > There are %s model with highest %s score" 
                % (num_candidates, cv_metric)
                , " within statistical uncertainties.")
            # Select preferred model
            id_selected_model = min(
                post_hoc_top.loc[post_hoc_top[5]==False][0].min(),
                post_hoc_top.loc[post_hoc_top[5]==False][1].min())
            print("\t > Model %s is selected by preference" % id_selected_model)
    else:
        print("ANOVA failed for %s. Model is selected by preferred order"
            % cv_metric)
        id_selected_model = 0
        cv_results['params'].iloc[id_selected_model]

    # Recreate hyperparameter combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Return parameters of the preferred model
    best_params = experiments[id_selected_model]
    print("Selected model:")
    print("\n".join(["\t- "+x+"="+str(best_params[x]) for x in best_params]))
    return {
        "f_stats": f_stats,
        "p_values": p_values,
        "tukey_all_results": post_hoc_results, 
        "tukey_top_results": post_hoc_top,
        "best_params": best_params}



            

