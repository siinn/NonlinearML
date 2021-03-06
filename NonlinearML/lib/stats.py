import itertools
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.weightstats import ttest_ind
#from statsmodels.stats.multicomp import pairwise_tukeyhsd

import NonlinearML.lib.io as io
import NonlinearML.lib.utils as utils


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
        io.message(" >> F-statistics = %s, p-value = %s" % (f_stat, p_value))
        io.message(" >> DOF1 (K-1) = %s, DOF2 (N-K) = %s" % (dof1, dof2))
    return round(f_stat, 6), round(p_value, 12)

def two_sample_ttest(df_stacked, p_thres):
    """ Perform two sample t-test."""
    # t-test
    test = ttest_ind(
        df_stacked.loc[df_stacked['model']==0]['perf'],
        df_stacked.loc[df_stacked['model']==1]['perf'])
    p_value = test[1]
    reject = p_value < p_thres
    return pd.DataFrame(
        data=[[0, 1, reject]],
        columns=['group1', 'group2', 'reject'])

def post_hoc_test(df_data, p_thres):
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
    # Perform t-test (2 samples) or Tukey test (> 2 samples)
    if df_stacked['model'].nunique() == 2:
        df_posthoc = two_sample_ttest(df_stacked, p_thres)
    else:
        post_hoc = MultiComparison(df_stacked['perf'], df_stacked['model'])
        df_posthoc = pd.DataFrame(
            data=post_hoc.tukeyhsd(alpha=p_thres).summary().data[1:],
            columns=post_hoc.tukeyhsd(alpha=p_thres).summary().data[0])
    return df_posthoc

def get_highest_correlated_metric(cv_results, cv_metric):
    """ Select metric with the highest correlation with top-bottom metric.
    Args:
        cv_results: Output of CV method (grid_search_purged_cv)
        cv_metrics: List of metrics that passed post-hoc test
    Returns:
        selected metric
    """
    io.message("Calculating correlation between metrics.")
    io.message("> metrics: %s" % cv_metric)
    # Set constants
    NUM_TAIL_CHAR = -4
    # If Top-Bottom in cv_metrics, it passed ANOVA. Just return itself
    if 'Top-Bottom' in cv_metric:
        return 'Top-Bottom'
    # Loop over metrics to correct values in validation set
    metric_val_names = [metric+"_val" for metric in cv_metric]
    metric_val_values = [metric+"_val_values" for metric in cv_metric]
    # Add Top-Bottom so that we can calculate correlation
    metric_val_names = metric_val_names + ['Top-Bottom_val']
    metric_val_values = metric_val_values + ['Top-Bottom_val_values']
    # Calculate correlation
    df_corr_val = pd.DataFrame()
    for name, values in zip(metric_val_names, metric_val_values):
        df_corr_val[name] = utils.expand_column(cv_results, values)\
            .stack().reset_index()[0]
    df_corr_val = df_corr_val.corr()
    return df_corr_val.drop('Top-Bottom_val')\
                      .sort_values('Top-Bottom_val', ascending=False)\
                      .index[0][:NUM_TAIL_CHAR]

def select_best_model_by_anova(
    cv_results, cv_metric, param_grid, p_thres, metric_selection=None):
    """ Select best model by performing ANOVA and post-hoc test.
    
    The preferred model is selected among the models within the statistical
    uncertainties of the model with the highest score.
    
    Minimum index is selected because cv_results is sorted by the
    preferred order of hyperparameters and values in param_grid.

    Args:
        cv_results: Output of CV method (grid_search_purged_cv)
        cv_metric: List of metric to use in model selction.
            Ex. ['f1-score', 'accuracy']
            Metric with ANOVA p-value < p_thres is used.
            If there are multiple metrics that satisfies condition, the
            metric is selcted by the order given in the list.
        param_grid: Hyperparameter grid used to return best hyperparameters
        p_thres: p-value threshold for ANOVA. ex. 0.05
        metric_selection: Set how metric should be selected.
            Available options: 
                corr: metric with highest correlation with Top-bottom
                    strategy is selected.
                p: metric with the lowest p-value is selected
                f: metric with the highest f-stat is selected
                combined: user-specified metrics are standardized and
                    averaged.
                None: metric is selected by pre-defined user preference.
    Returns:
        f_stats: f-statistics of all models calculated using each metric
        p_values: p-values of all models calculated using each metric
        post_hoc_results: results of post-hoc tests 
        best_params: best parameter selected
        id_selected_model: ID of the selected model,

    """
    def _convert_to_float(x):
        """ Attempts to convert to float if possible."""
        try:
            return float(x)
        except ValueError:
            return x

    # Extract list of metrics
    metric_names = [
        metric[:-11] for metric in cv_results.columns if 'val_values' in metric]

    # Perform ANOVA and post hoc test on each metrics
    f_stats = {}
    p_values = {}
    post_hoc_results = {}
    post_hoc_top = {}

    # Loop over each metric
    io.title("ANOVA and post hoc test on model performance")
    io.message("Performing ANOVA test..")
    io.message(" > p-value threshold: %s" % p_thres)
    for metric in metric_names:
        values = metric+"_val_values"

        # Extract values of CV results
        model_perf = utils.expand_column(cv_results, values)

        # Perform one way ANOVA
        f_stats[metric], p_values[metric] = anova_test(model_perf)
        io.message("\t> %s: p-value = %s" % (metric, p_values[metric]))

        # If p-value is less than the threshold, perform post hoc test.
        if p_values[metric] < p_thres:
            # If post-hoc test fails for all pair, discard from post hoc results
            if post_hoc_test(model_perf, p_thres)['reject'].sum() > 0:
                post_hoc_results[metric] = post_hoc_test(model_perf, p_thres)
            else:
                io.message("\t> (%s failed post-hoc test)" % (metric))

    # Select metric with associated p-value < p_thres
    cv_metric = [x for x in cv_metric if x in post_hoc_results.keys()]

    # If one of the specified metrics passed ANOVA, perform post hoc test
    if len(cv_metric) > 0:
        # Select metric
        if metric_selection=='corr':
            io.message("Metric selection: By correlation with Top-Bottom strategy")
            selected_metric = get_highest_correlated_metric(cv_results, cv_metric)
        elif metric_selection=='p':
            io.message("Metric selection: By p-value")
            selected_metric = sorted(
                p_values.items(), key=lambda x:x[1])[0][0]
        elif metric_selection=='f':
            io.message("Metric selection: By f-stat")
            selected_metric = sorted(
                f_stats.items(), key=lambda x:x[1], reverse=True)[0][0]
        elif metric_selection=='combined':
            io.message("Metric selection: By combined metric")
            if not 'combined_zscore' in post_hoc_results.keys():
                io.message("> Combined score didn't pass ANOVA test.")
                io.message("> Metric is selected by pre-defined user preference.")
                selected_metric = cv_metric[0]
            else:
                selected_metric = 'combined_zscore'
        else:
            io.message("Metric selection: By pre-defined user preference")
            selected_metric = cv_metric[0]
        io.message("> %s is selected as metric." %selected_metric)

        # Index of the model with the highest score
        id_max = cv_results[selected_metric+"_val_mean"].idxmax()
        io.message("\t> Model %s has the highest score" % id_max)

        # Filter result by the model with the highest score
        post_hoc_top = post_hoc_results[selected_metric].loc[
            (post_hoc_results[selected_metric]['group1']==id_max) |
            (post_hoc_results[selected_metric]['group2']==id_max)]
        # Check if there are multiple models within statistical uncertainties
        num_candidates = post_hoc_top.loc[post_hoc_top['reject']==False]\
            .shape[0] + 1 # Adding 1 to include the highest model
        if num_candidates < 2:
            io.message("\t > There is only one model with highest %s score." % selected_metric)
            id_selected_model = id_max
        else:
            io.message("\t > There are %s models with highest %s score"\
                % (num_candidates, selected_metric)\
                + " within statistical uncertainties.")
            # Select preferred model
            id_selected_model = min(
                post_hoc_top.loc[post_hoc_top['reject']==False]['group1'].min(),
                post_hoc_top.loc[post_hoc_top['reject']==False]['group2'].min())
            io.message("\t > Model %s is selected by preference" % id_selected_model)
    else:
        io.message("ANOVA failed in all metrics. Model is selected by preferred order")
        id_selected_model = 0
        cv_results['params'].iloc[id_selected_model]

    # Recreate hyperparameter combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Return parameters of the preferred model
    best_params = experiments[id_selected_model]
    io.message("Selected model:")
    io.message(["\t- "+x+"="+str(best_params[x]) for x in best_params])

    return {
        "f_stats": f_stats,
        "p_values": p_values,
        "tukey_all_results": post_hoc_results, 
        "tukey_top_results": post_hoc_top,
        "best_params": best_params,
        "id_selected_model": id_selected_model,
        }



            

