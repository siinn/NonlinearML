# Import custom libraries
import numpy as np

import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.utils as utils
import NonlinearML.lib.stats as stats

import NonlinearML.plot.decision_boundary as plot_db
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.cross_validation as plot_cv


def decision_boundary2D(
    config,
    df_train, df_test,
    model, model_str, param_grid,
    best_params={},
    cv_study=True, calculate_return=True, plot_decision_boundary=True,
    cv_hist_n_bins=10, cv_hist_figsize=(18, 10), cv_hist_alpha=0.6,
    cv_box_figsize=(18,10), cv_box_color="#3399FF",
    db_res=0.01, db_figsize=(10,8), db_xlim=(-3,3),
    db_ylim=(-3,3), db_annot_x=0.02, db_annot_y=0.98,
    return_figsize=(8,6), return_train_ylim=(-1,7), return_test_ylim=(-1,5),
    return_diff_train_ylim=(-1,7), return_diff_test_ylim=(-1,5)):
    """
    Args:
        config: Global configuration passed as dictionary
        df_train, df_test: Dataframe in the following format.
            -----------------------------------------------------------------
            Date_column, feature_x, feature_y, label_reg, label_cla, label_fm
            -----------------------------------------------------------------
            2017-01-01  0.01    0.23    -0.24   1.0 0.32
        model: ML model object with the following methods:
            .predict, .fit, .set_params
        model_str: String represents model name. Ex. 'lr' or 'xgb'
        param_grid: Dictionary of hyperparameter sets.
            Example: {'C':[0,1], 'penalty':['l2']

        best_params: Dictionary of best hyperparameters. If this is specified,
            this function will not perform grid search. Instead, it will use
            the best params for model fitting.

        cv_study: Perform study on cross-validation if True
        calculate_return: Calculate cumulative return if True
        plot_decision_boundary: Plot decision boundary of the best model if True


        Others: parameters for nested functions.

    Returns: Dictionary containing the following results
        'cv_results': Model performance obtained from cross-validation method
        'anova_results': Results from anova test on models with different
            hyperparameters.
        'pred_train': Prediction made by the best model on train dataset.
        'pred_test': Prediction made by the best model on test dataset.
        'cum_return_train': Cumulative return calculated from train dataset.
        'cum_return_test': Cumulative return calculated from test dataset.
        'model': Trained model with the best hyperparameter set.

    """
    print("Running %s" % model_str)

    # Set output path for this model
    output_path = config['output_path'] + model_str + '/'

    # Set features of interest
    features = [config['feature_x'], config['feature_y']]
    
    #-----------------------------------------------------------------------
    # Perform hyperparameter search using cross-validation
    #-----------------------------------------------------------------------
    if not best_params:
        cv_results = cv.grid_search(
            df_train=df_train,
            model=model,
            param_grid=param_grid,
            metric=config['cv_metric'],
            n_epoch=config['n_epoch'], subsample=config['subsample'],
            features=features, label=config['label_cla'],
            k=config['k'], purge_length=config['purge_length'],
            output_path=output_path+"cross_validation/",
            verbose=False)
        # Perform ANOVA to select best model
        anova_results = stats.select_best_model_by_anova(
            cv_results=cv_results,
            cv_metric=config['cv_metric'],
            param_grid=param_grid, p_thres=config['p_thres'])

        # Get best parameters
        best_params = anova_results['best_params']
    
        #-----------------------------------------------------------------------
        # Cross-validation study
        #-----------------------------------------------------------------------
        if cv_study:
            # Plot distribution of cross-validation results
            plot_cv.plot_cv_dist(
                cv_results,
                n_bins=cv_hist_n_bins, x_range=None, legend_loc=None,
                legend_box=(1, 1), figsize=cv_hist_figsize, alpha=cv_hist_alpha,
                hist_type='stepfilled', edgecolor='black',
                filename=output_path+"cross_validation/cv_hist")
            plot_cv.plot_cv_box(
                cv_results,
                filename=output_path+"cross_validation/cv_box",
                cv_metric=None, figsize=cv_box_figsize, color=cv_box_color)
            
            # Plot decision boundaries of all hyperparameter sets
            plot_db.decision_boundary_multiple_hparmas(
                param_grid=param_grid, label=config['label_cla'], model=model,
                df=df_train, features=features, h=db_res,
                x_label=config['feature_x'], y_label=config['feature_y'],
                #vlines=tertile_boundary[feature_x],
                #hlines=tertile_boundary[feature_y],
                colors=config['db_colors'], xlim=db_xlim, ylim=db_ylim,
                figsize=db_figsize,
                ticks=sorted(np.arange(config['n_classes'])),
                filename=output_path+"decision_boundary/db",
                annot={
                    'text':str(best_params).strip('{}')\
                        .replace('\'','').replace(',','\n')\
                        .replace('\n ', '\n'),
                    'x':db_annot_x, 'y':db_annot_y},
                )
    
    else:
        # If grid search is not performed, set results to be an empty dictionary
        cv_results={}
        anova_results={}
    
    #---------------------------------------------------------------------------
    # Prediction
    #---------------------------------------------------------------------------
    # Make prediction using best parameters
    pred_train, pred_test, model = utils.predict(
            model=model.set_params(**best_params),
            df_train=df_train, df_test=df_test, features=features,
            label_cla=config['label_cla'], label_fm=config['label_fm'],
            time=config['date_column'])

    #---------------------------------------------------------------------------
    # Cumulative return
    #---------------------------------------------------------------------------
    if calculate_return:
        # Calculate cumulative return using trained model
        df_cum_train, df_cum_test = utils.calculate_cum_return(
                pred_train=pred_train, pred_test=pred_test, 
                label_fm=config['label_fm'], time=config['date_column'])
        
        # Make cumulative return plot
        plot_backtest.plot_cumulative_return(
            df_cum_train, df_cum_test, config['label_reg'],
            group_label=config['class_label'],
            figsize=return_figsize,
            filename=output_path+"cum_return/return_by_group",
            kwargs_train={'ylim':return_train_ylim},
            kwargs_test={'ylim':return_test_ylim})
        plot_backtest.plot_cumulative_return_diff(
            list_cum_returns=[df_cum_test],
            return_label=sorted(np.arange(config['n_classes'])),
            list_labels=[model_str], label_reg=config['label_reg'],
            figsize=return_figsize,
            filename=output_path+"cum_return/return_diff_group",
            kwargs_train={'ylim':return_diff_train_ylim},
            kwargs_test={'ylim':return_diff_test_ylim})
    else:
        # If cumulative returns are not calculated, create dummy results
            df_cum_train = None
            df_cum_test = None


    #---------------------------------------------------------------------------
    # Decision boundary
    #---------------------------------------------------------------------------
    if plot_decision_boundary:
        # Plot decision boundary of the best model.
        plot_db.decision_boundary(
            model=model, df=df_test, features=features, h=db_res,
            x_label=config['feature_x'], y_label=config['feature_y'],
            #vlines=tertile_boundary[feature_x],
            #hlines=tertile_boundary[feature_y],
            colors=config['db_colors'],
            xlim=db_xlim, ylim=db_ylim, figsize=db_figsize,
            ticks=sorted(np.arange(config['n_classes']), reverse=True),
            annot={
                'text':str(best_params).strip('{}')\
                    .replace('\'','').replace(',','\n').replace('\n ', '\n'),
                'x':db_annot_x, 'y':db_annot_y},
            filename=output_path+"decision_boundary/db_best_model")


    return {
        'cv_results': cv_results, 'anova_results': anova_results,
        'pred_train': pred_train, 'pred_test': pred_test,
        'cum_return_train': df_cum_train, 'cum_return_test': df_cum_test,
        'model': model}
            

