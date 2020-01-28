# Import custom libraries
import numpy as np
import pandas as pd
#import logging


import NonlinearML.lib.backtest as backtest
import NonlinearML.lib.cross_validation as cv
import NonlinearML.lib.io as io
import NonlinearML.lib.stats as stats
import NonlinearML.lib.utils as utils

import NonlinearML.plot.plot as plot
import NonlinearML.plot.decision_boundary as plot_db
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.cross_validation as plot_cv


def regression_surface2D(
    config,
    df_train, df_test,
    model, model_str, param_grid, best_params={},
    read_last=False, cv_study=None, run_backtest=True, model_evaluation=True,
    plot_decision_boundary=True, plot_residual=True, save_csv=True,
    cv_hist_n_bins=10, cv_hist_figsize=(18, 10), cv_hist_alpha=0.8,
    cv_box_figsize=(18,10), cv_box_color="#3399FF",
    return_figsize=(8,6), return_train_ylim=(-1,20), return_test_ylim=(-1,1),
    return_diff_test_ylim=(-1,5), verbose=False):
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
        best_params: Dictionary of best hyperparameters.
            Override best parameters found by grid search
        read_last: If True, it reads cv_results from local path. If False, it
            performs grid search
        cv_study: Perform study on cross-validation if True
        run_backtest: Calculate cumulative return, annual return, and IR if True
        plot_decision_boundary: Plot decision boundary of the best model if True
        plot_residual: Examine distribution of target, prediction, and residual
            if True
        save_csv: Save all results as csv
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
    # Set output path for this model
    output_path = config['output_path'] + model_str + '/'

    # Set logging configuration
    io.setConfig(path=output_path, filename="log.txt")
    io.title('Running two factor regression with factors:')
    io.message(' > feature x: %s' % config['feature_x'])
    io.message(' > feature y: %s' % config['feature_y'])
    io.message(" > Running %s" % model_str)

    # Set features of interest
    features = [config['feature_x'], config['feature_y']]

    # Set prediction label
    label=config['label_reg']
    
    #---------------------------------------------------------------------------
    # Perform hyperparameter search using cross-validation
    #---------------------------------------------------------------------------
    if read_last:
        io.title("Import CV results")
        io.message("Grid search is set to False. Reading CV results from:")
        cv_path = output_path + "csv/cv_results.csv"
        io.message(" > " + cv_path)

        # Read CV results from local file
        try:
            cv_results = pd.read_csv(cv_path, index_col='Unnamed: 0')
        except IOError:
            io.error("CV results is not available at %s" % cv_path)
    else:
        cv_results = cv.grid_search(
            df_train=df_train,
            model=model, model_type='reg',
            param_grid=param_grid,
            n_epoch=config['n_epoch'], subsample=config['subsample'],
            features=features, label=label, label_fm=config['label_fm'],
            date_column=config['date_column'],
            train_from_future=config['train_from_future'],
            k=config['k'], purge_length=config['purge_length'],
            cv_metric=config['cv_metric'],
            output_path=output_path+"cross_validation/",
            rank_n_bins=config['rank_n_bins'], rank_label=config['rank_label'],
            rank_top=config['rank_top'], rank_bottom=config['rank_bottom'],
            force_val_length=config['force_val_length'],
            verbose=verbose)


    #---------------------------------------------------------------------------
    # Perform ANOVA to select best model
    #---------------------------------------------------------------------------
    anova_results = stats.select_best_model_by_anova(
        cv_results=cv_results,
        cv_metric=config['cv_metric'],
        param_grid=param_grid, p_thres=config['p_thres'])

    # Override best parameters with the specified set.
    if not best_params:
        best_params = anova_results['best_params']

    
    #---------------------------------------------------------------------------
    # Cross-validation study
    #---------------------------------------------------------------------------
    if cv_study:
        # Plot distribution of cross-validation results
        plot_cv.plot_cv_dist(
            cv_results,
            n_bins=cv_hist_n_bins, x_range=None, legend_loc=None,
            legend_box=(1, 1), figsize=cv_hist_figsize, alpha=cv_hist_alpha,
            hist_type='stepfilled', edgecolor='black',
            filename=output_path+"cross_validation/hist/cv_hist")

        plot_cv.plot_cv_box(
            cv_results,
            filename=output_path+"cross_validation/box/cv_box",
            figsize=cv_box_figsize, color=cv_box_color)

        plot_cv.plot_cv_line(
            cv_results, filename=output_path+"cross_validation/line/cv_line",
            marker='.', markersize=20)

        plot_cv.plot_cv_correlation_heatmap(
            cv_results, 
            filename=output_path+"cross_validation/correlation/cv_corr")

        plot_cv.plot_cv_correlation_scatter(
            cv_results, 
            filename=output_path+"cross_validation/correlation/cv_corr")


        # Plot decision boundaries of all hyperparameter sets
        plot_db.decision_boundary_multiple_hparmas(
            param_grid=param_grid,
            label=label,
            label_cla=config['label_cla'],
            db_annot_x=config['db_annot_x'],
            db_annot_y=config['db_annot_y'],
            h=config['db_res'], figsize=config['db_figsize'],
            x_label=config['feature_x'], y_label=config['feature_y'],
            colors=config['db_colors'],
            xlim=config['db_xlim'], ylim=config['db_ylim'],
            vmin=config['db_vmin'], vmax=config['db_vmax'],
            #ticks=sorted(list(config['rank_label'].keys())),
            colorbar=True, ticks=None,
            scatter=True, subsample=0.01,
            scatter_legend=False, colors_scatter=config['db_colors_scatter'],
            dist=True, nbins=config['db_nbins'],
            model=model,
            df=df_train,
            features=features, 
            filename=output_path+"decision_boundary/overlay_db")
    
    #---------------------------------------------------------------------------
    # Prediction
    #---------------------------------------------------------------------------
    # Make prediction using best parameters
    if plot_decision_boundary or run_backtest:
        pred_train, pred_test, model = utils.predict(
                model=model.set_params(**best_params),
                df_train=df_train, df_test=df_test, features=features,
                date_column=config['date_column'],
                label=label,
                cols=[config['label_fm'], config['label_reg']],
                expand_window=config['expand_training_window']
                )

    else:
        pred_train = pred_test = model = None

    #---------------------------------------------------------------------------
    # Rank prediction by each month
    #---------------------------------------------------------------------------
    pred_train, pred_test = utils.rank_prediction_monthly(
        pred_train=pred_train, pred_test=pred_test,
        config=config, col_pred="pred")


    #---------------------------------------------------------------------------
    # Model evaluation
    #---------------------------------------------------------------------------
    if model_evaluation:
        model_evaluation_train  = {}
        model_evaluation_test  = {}
    
        # Evaluate by standard metrics
        model_evaluation_train = cv.evaluate_regressor(
            pred_train[config['label_reg']], pred_train['pred'],
            model_evaluation_train, epsilon=1e-7)
    
        model_evaluation_test = cv.evaluate_regressor(
            pred_test[config['label_reg']], pred_test['pred'],
            model_evaluation_test, epsilon=1e-7)
    
        # Evaluate by top-bottom strategy
        if 'rank_n_bins' in config:
            model_evaluation_train = cv.evaluate_top_bottom_strategy(
                df=pred_train,
                date_column=config['date_column'],
                label_fm=config['label_fm'], y_pred=pred_train['pred'],
                rank_n_bins=config['rank_n_bins'],
                rank_label=config['rank_label'],
                rank_top=config['rank_top'], rank_bottom=config['rank_bottom'],
                results=model_evaluation_train)
        
            model_evaluation_test = cv.evaluate_top_bottom_strategy(
                df=pred_test,
                date_column=config['date_column'],
                label_fm=config['label_fm'], y_pred=pred_test['pred'],
                rank_n_bins=config['rank_n_bins'],
                rank_label=config['rank_label'],
                rank_top=config['rank_top'], rank_bottom=config['rank_bottom'],
                results=model_evaluation_test)
    
    #---------------------------------------------------------------------------
    # Backtesting
    #---------------------------------------------------------------------------
    if run_backtest:

        # Calculate cumulative return using trained model
        df_backtest_train, df_backtest_test = backtest.perform_backtest(
                pred_train=pred_train, pred_test=pred_test, 
                col_pred='pred_rank',
                list_class=list(config['rank_label'].keys()),
                label_fm=config['label_fm'], time=config['date_column'])

        # Calculate diff. in cumulative return, annual return, and IR
        df_diff_train = backtest.calculate_diff_IR(
            df=df_backtest_train, 
            top=config['rank_top'], bottom=config['rank_bottom'],
            class_label=config['rank_label'],
            class_reg=config['label_fm'],
            time=config['date_column'],
            col_pred='pred_rank')
        df_diff_test = backtest.calculate_diff_IR(
            df=df_backtest_test, 
            top=config['rank_top'], bottom=config['rank_bottom'],
            class_label=config['rank_label'],
            class_reg=config['label_fm'],
            time=config['date_column'],
            col_pred='pred_rank')
        

        # Make cumulative return plot
        plot_backtest.plot_cumulative_return(
            df_backtest_train, df_backtest_test, config['label_fm'],
            group_label=config['rank_label'],
            figsize=return_figsize,
            filename=output_path+"cum_return/return_by_group",
            date_column=config['date_column'],
            train_ylim=return_train_ylim,
            test_ylim=return_test_ylim,
            col_pred='pred_rank')
        plot_backtest.plot_cumulative_return_diff(
            list_cum_returns=[df_backtest_test],
            top=config['rank_top'], bottom=config['rank_bottom'],
            class_label=config['rank_label'],
            list_labels=[model_str], label_reg=config['label_fm'],
            figsize=return_figsize,
            date_column=config['date_column'],
            filename=output_path+"cum_return/return_diff_group",
            ylim=return_diff_test_ylim,
            col_pred='pred_rank')
    else:
        # If cumulative returns are not calculated, create dummy results
        df_backtest_train = None
        df_backtest_test = None
        df_diff_train = None
        df_diff_test = None


    #---------------------------------------------------------------------------
    # Decision boundary
    #---------------------------------------------------------------------------
    if plot_decision_boundary:

        # Plot decision boundary of the best model with scattered plot.
        plot_db.decision_boundary(
            model=model, df=df_train, features=features, h=config['db_res'],
            x_label=config['feature_x'], y_label=config['feature_y'],
            colors=config['db_colors'],
            xlim=config['db_xlim'], ylim=config['db_ylim'],
            vmin=config['db_vmin'], vmax=config['db_vmax'],
            figsize=config['db_figsize'],
            colorbar=True, ticks=None,
            annot={
                'text':utils.get_param_string(best_params).strip('{}')\
                    .replace('\'','').replace(',','\n').replace('\n ', '\n'),
                'x':config['db_annot_x'], 'y':config['db_annot_y']},
            scatter=True, subsample=0.01, label_cla=config['label_cla'],
            scatter_legend=False, colors_scatter=config['db_colors_scatter'],
            dist=True, nbins=config['db_nbins'],
            filename=output_path+"decision_boundary/overlay_db_best_model")


        # Plot decision boundary of the best model without scattered plot.
        plot_db.decision_boundary(
            model=model, df=df_train, features=features, h=config['db_res'],
            x_label=config['feature_x'], y_label=config['feature_y'],
            colors=config['db_colors'],
            xlim=config['db_xlim'], ylim=config['db_ylim'],
            vmin=config['db_vmin'], vmax=config['db_vmax'],
            figsize=config['db_figsize'],
            colorbar=True, ticks=None,
            annot={
                'text':utils.get_param_string(best_params).strip('{}')\
                    .replace('\'','').replace(',','\n').replace('\n ', '\n'),
                'x':config['db_annot_x'], 'y':config['db_annot_y']},
            scatter=False, subsample=0.01, label_cla=config['label_cla'],
            scatter_legend=False, colors_scatter=config['db_colors_scatter'],
            dist=True, nbins=config['db_nbins'],
            filename=output_path+"decision_boundary/db_best_model")

    #---------------------------------------------------------------------------
    # Examine residual
    #---------------------------------------------------------------------------
    if plot_residual:
        # Plot distribution of residual
        # Calculate residual
        res_train = pred_train[config['label_reg']] - pred_train['pred']
        res_test = pred_test[config['label_reg']] - pred_test['pred']
        
        # Plot distribution of prediction and target
        plot.plot_dist_hue(
            df= pd.concat(
                [res_train, res_test], keys=['Train', 'Test'], axis=1)\
                .stack()\
                .reset_index(level=-1)\
                .rename(
                    {0:'Residual (%s - Prediction)' %config['label_reg']},
                    axis=1),
            x='Residual (%s - Prediction)' %config['label_reg'], hue="level_1",
            hue_str={'Train':'Train (normalized)', 'Test':'Test (normalized)'},
            hist_type='step', ylabel='Samples', norm=True,
            n_bins=config['residual_n_bins'],
            figsize=(8,5), filename=output_path+'residual/res_dist')


        # Plot distribution of prediction and target
        plot.plot_dist_hue(
            df=pred_train[[config['label_reg'], 'pred']].stack()\
                .reset_index(level=-1).rename({0:config['label_reg']}, axis=1),
            x=config['label_reg'], hue="level_1",
            hue_str={config['label_reg']:'True', 'pred':'Prediction'},
            hist_type='step', ylabel='Samples', norm=False,
            n_bins=config['residual_n_bins'],
            figsize=(8,5), filename=output_path+'residual/dist_train')

        plot.plot_dist_hue(
            df=pred_test[[config['label_reg'], 'pred']].stack()\
                .reset_index(level=-1).rename({0:config['label_reg']}, axis=1),
            x=config['label_reg'], hue="level_1",
            hue_str={config['label_reg']:'True', 'pred':'Prediction'},
            hist_type='step', ylabel='Samples', norm=False,
            n_bins=config['residual_n_bins'],
            figsize=(8,5), filename=output_path+'residual/dist_test')

        # Plot prediction vs target
        plot.plot_reg(
            df=pred_train, x=config['label_reg'], y='pred',
            x_label=config['label_reg'], y_label='Prediction',
            figsize=(10, 10), filename=output_path+'residual/res_scatter_train',
            fit_reg=True, scatter_kws={'linewidth':1, 's':1, 'color':'dodgerblue'},
            line_kws={'color':'crimson'})

        plot.plot_reg(
            df=pred_test, x=config['label_reg'], y='pred',
            x_label=config['label_reg'], y_label='Prediction',
            figsize=(10, 10), filename=output_path+'residual/res_scatter_test',
            fit_reg=True, scatter_kws={'linewidth':1, 's':1, 'color':'slategray'},
            line_kws={'color':'crimson'})



    #---------------------------------------------------------------------------
    # Save output as csv
    #---------------------------------------------------------------------------
    if save_csv:
        utils.create_folder(output_path+'csv/summary.csv')
        # Save the summary
        if not read_last:
            cv_results.to_csv(output_path+'csv/cv_results.csv')
        # Save
        if run_backtest:
            try:
                df_backtest_train.to_csv(output_path+'csv/cum_return_train.csv')
                df_backtest_test.to_csv(output_path+'csv/cum_return_test.csv')
                df_diff_train.to_csv(output_path+'csv/backtest_diff_train.csv')
                df_diff_test.to_csv(output_path+'csv/backtest_diff_test.csv')
            except IOError:
                io.error("Cannot save backtesting results.")
        # Save ANOVA results
        try:
            for key in anova_results:
                if key in ['f_stats', 'p_values', 'best_params']:
                    pd.DataFrame.from_dict(anova_results[key], orient='index')\
                        .to_csv(output_path+'csv/anova_%s.csv' %key)
            # All tukey test results
            for metric in anova_results['tukey_all_results'].keys():
                if type(anova_results['tukey_all_results'][metric]) == pd.DataFrame:
                    anova_results['tukey_all_results'][metric].to_csv(
                            output_path+'csv/tukey_all_results_%s.csv' % metric)
            # Tukey test results related to the top performing model
            if type(anova_results['tukey_top_results']) == pd.DataFrame:
                anova_results['tukey_top_results'].to_csv(
                    output_path+'csv/tukey_top_results.csv')
            # Save ID of the selected model
            pd.DataFrame(
                data=[anova_results['id_selected_model']],
                columns=['id_selected_model']).to_csv(
                    output_path+'csv/id_selected_model.csv')
        except IOError:
            io.error("Cannot save ANOVA results.")
        # Save predictions
        if run_backtest or plot_decision_boundary:
            try:
                pred_train.to_csv(output_path+'csv/pred_train.csv')
                pred_test.to_csv(output_path+'csv/pred_test.csv')
            except IOError:
                io.error("Cannot save predictions.")
        # Save model evaluation
        if model_evaluation:
            pd.DataFrame([
                model_evaluation_train, model_evaluation_test],
                index=['Train', 'Test'])\
            .applymap(lambda x:x[0])\
            .to_csv(output_path+'csv/model_evaluation.csv')

    io.message("Successfully completed all tasks!")

    return {
        'cv_results': cv_results, 'anova_results': anova_results,
        'pred_train': pred_train, 'pred_test': pred_test,
        'model_evaluation_train': model_evaluation_train,
        'model_evaluation_test': model_evaluation_test,
        'cum_return_train': df_backtest_train, 'cum_return_test': df_backtest_test,
        'model': model}
            


def regression_surface2D_residual(
    config,
    df_train, df_test,
    model, model_str, param_grid, best_params={},
    read_last=False, cv_study=None, run_backtest=True, model_evaluation=True,
    plot_decision_boundary=True, plot_residual=True, save_csv=True,
    cv_hist_n_bins=10, cv_hist_figsize=(12, 8), cv_hist_alpha=0.6,
    cv_box_figsize=(18,10), cv_box_color="#3399FF",
    return_figsize=(8,6), return_train_ylim=(-1,20), return_test_ylim=(-1,1),
    return_diff_test_ylim=(-1,5), verbose=False):
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
        best_params: Dictionary of best hyperparameters.
            Override best parameters found by grid search
        read_last: If True, it reads cv_results from local path. If False, it
            performs grid search
        cv_study: Perform study on cross-validation if True
        run_backtest: Calculate cumulative return, annual return, and IR if True
        plot_decision_boundary: Plot decision boundary of the best model if True
        plot_residual: Examine distribution of target, prediction, and residual
            if True
        save_csv: Save all results as csv
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
    # Set output path for this model
    output_path = config['output_path'] + model_str + '/'

    # Set logging configuration
    io.setConfig(path=output_path, filename="log.txt")
    io.title('Running two factor regression with factors:')
    io.message(' > feature x: %s' % config['feature_x'])
    io.message(' > feature y: %s' % config['feature_y'])
    io.message(" > Running %s" % model_str)

    # Set features of interest
    features = [config['feature_x'], config['feature_y']]

    # Set prediction label
    label=config['label_reg']
    
    #---------------------------------------------------------------------------
    # Perform hyperparameter search using cross-validation
    #---------------------------------------------------------------------------
    if read_last:
        io.title("Import CV results")
        io.message("Grid search is set to False. Reading CV results from:")
        cv_path = output_path + "csv/cv_results.csv"
        io.message(" > " + cv_path)

        # Read CV results from local file
        try:
            cv_results = pd.read_csv(cv_path, index_col='Unnamed: 0')
        except IOError:
            io.error("CV results is not available at %s" % cv_path)
    else:
        cv_results = cv.grid_search(
            df_train=df_train,
            model=model, model_type='reg',
            param_grid=param_grid,
            n_epoch=config['n_epoch'], subsample=config['subsample'],
            features=features, label=label, label_fm=config['label_fm'],
            date_column=config['date_column'],
            train_from_future=config['train_from_future'],
            k=config['k'], purge_length=config['purge_length'],
            cv_metric=config['cv_metric'],
            output_path=output_path+"cross_validation/",
            rank_n_bins=config['rank_n_bins'], rank_label=config['rank_label'],
            rank_top=config['rank_top'], rank_bottom=config['rank_bottom'],
            force_val_length=config['force_val_length'],
            verbose=verbose)


    #---------------------------------------------------------------------------
    # Perform ANOVA to select best model
    #---------------------------------------------------------------------------
    anova_results = stats.select_best_model_by_anova(
        cv_results=cv_results,
        cv_metric=config['cv_metric'],
        param_grid=param_grid, p_thres=config['p_thres'])

    # Override best parameters with the specified set.
    if not best_params:
        best_params = anova_results['best_params']

    
    #---------------------------------------------------------------------------
    # Cross-validation study
    #---------------------------------------------------------------------------
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
            figsize=cv_box_figsize, color=cv_box_color)

        plot_cv.plot_cv_line(
            cv_results, filename=output_path+"cross_validation/cv_line",
            marker='.', markersize=20)

        # Plot decision boundaries of all hyperparameter sets
        plot_db.decision_boundary_multiple_hparmas(
            param_grid=param_grid,
            label=label,
            label_cla=config['label_cla'],
            db_annot_x=config['db_annot_x'],
            db_annot_y=config['db_annot_y'],
            vmin=config['db_vmin'], vmax=config['db_vmax'],
            h=config['db_res'], figsize=config['db_figsize'],
            x_label=config['feature_x'], y_label=config['feature_y'],
            colors=config['db_colors'],
            xlim=config['db_xlim'], ylim=config['db_ylim'],
            #ticks=sorted(list(config['rank_label'].keys())),
            colorbar=True, ticks=None,
            scatter=True, subsample=0.01,
            scatter_legend=False, colors_scatter=config['db_colors_scatter'],
            dist=True, nbins=config['db_nbins'],
            model=model,
            df=df_train,
            features=features, 
            filename=output_path+"decision_boundary/overlay_db")
    
    #---------------------------------------------------------------------------
    # Prediction
    #---------------------------------------------------------------------------
    # Make prediction using best parameters
    if plot_decision_boundary or run_backtest:
        pred_train, pred_test, model = utils.predict(
                model=model.set_params(**best_params),
                df_train=df_train, df_test=df_test, features=features,
                date_column=config['date_column'],
                label=label,
                cols=[
                    config['label_reg'],
                    config['label_fm'],
                    config['label_edge']])
    else:
        pred_train = pred_test = model = None

    #---------------------------------------------------------------------------
    # Rank prediction by each month
    #---------------------------------------------------------------------------
    # Rank predicted residual
    pred_train, pred_test = utils.rank_prediction_monthly(
        pred_train=pred_train, pred_test=pred_test,
        config=config, col_pred="pred")

    # Recover returnby summing edge and residual. Rank predicted edge + residual
    pred_train['pred_return'] = \
        pred_train[config['label_edge']] + pred_train['pred']
    pred_test['pred_return'] = \
        pred_test[config['label_edge']] + pred_test['pred']
    pred_train, pred_test = utils.rank_prediction_monthly(
        pred_train=pred_train, pred_test=pred_test,
        config=config, col_pred="pred_return")
        
    #---------------------------------------------------------------------------
    # Model evaluation
    #---------------------------------------------------------------------------
    if model_evaluation:
        model_evaluation_train  = {}
        model_evaluation_test  = {}

        # Evaluate by standard metrics
        model_evaluation_train = cv.evaluate_regressor(
            pred_train[config['label_reg']], pred_train['pred'],
            model_evaluation_train, epsilon=1e-7)

        model_evaluation_test = cv.evaluate_regressor(
            pred_test[config['label_reg']], pred_test['pred'],
            model_evaluation_test, epsilon=1e-7)

        # Evaluate by top-bottom strategy
        if 'rank_n_bins' in config:
            model_evaluation_train = cv.evaluate_top_bottom_strategy(
                df=pred_train,
                date_column=config['date_column'],
                label_fm=config['label_fm'], y_pred=pred_train['pred_return'],
                rank_n_bins=config['rank_n_bins'],
                rank_label=config['rank_label'],
                rank_top=config['rank_top'], rank_bottom=config['rank_bottom'],
                results=model_evaluation_train)

            model_evaluation_test = cv.evaluate_top_bottom_strategy(
                df=pred_test,
                date_column=config['date_column'],
                label_fm=config['label_fm'], y_pred=pred_test['pred_return'],
                rank_n_bins=config['rank_n_bins'],
                rank_label=config['rank_label'],
                rank_top=config['rank_top'], rank_bottom=config['rank_bottom'],
                results=model_evaluation_test)
    

    #---------------------------------------------------------------------------
    # Backtesting
    #---------------------------------------------------------------------------
    if run_backtest:
        # Calculate cumulative return using trained model
        df_backtest_train, df_backtest_test = backtest.perform_backtest(
                pred_train=pred_train, pred_test=pred_test, 
                col_pred='pred_return_rank',
                list_class=list(config['rank_label'].keys()),
                label_fm=config['label_fm'], time=config['date_column'])

        # Calculate diff. in cumulative return, annual return, and IR
        df_diff_train = backtest.calculate_diff_IR(
            df=df_backtest_train, 
            top=config['rank_top'], bottom=config['rank_bottom'],
            class_label=config['rank_label'],
            class_reg=config['label_fm'],
            time=config['date_column'],
            col_pred='pred_return_rank')
        df_diff_test = backtest.calculate_diff_IR(
            df=df_backtest_test, 
            top=config['rank_top'], bottom=config['rank_bottom'],
            class_label=config['rank_label'],
            class_reg=config['label_fm'],
            time=config['date_column'],
            col_pred='pred_return_rank')
        

        # Make cumulative return plot
        plot_backtest.plot_cumulative_return(
            df_backtest_train, df_backtest_test, config['label_fm'],
            group_label=config['rank_label'],
            figsize=return_figsize,
            filename=output_path+"cum_return/return_by_group",
            date_column=config['date_column'],
            train_ylim=return_train_ylim,
            test_ylim=return_test_ylim,
            col_pred='pred_return_rank')
        plot_backtest.plot_cumulative_return_diff(
            list_cum_returns=[df_backtest_test],
            top=config['rank_top'], bottom=config['rank_bottom'],
            class_label=config['rank_label'],
            list_labels=[model_str], label_reg=config['label_fm'],
            figsize=return_figsize,
            date_column=config['date_column'],
            filename=output_path+"cum_return/return_diff_group",
            ylim=return_diff_test_ylim,
            col_pred='pred_return_rank')
    else:
        # If cumulative returns are not calculated, create dummy results
        df_backtest_train = None
        df_backtest_test = None
        df_diff_train = None
        df_diff_test = None


    #---------------------------------------------------------------------------
    # Decision boundary
    #---------------------------------------------------------------------------
    if plot_decision_boundary:

        # Plot decision boundary of the best model with scattered plot.
        plot_db.decision_boundary(
            model=model, df=df_train, features=features, h=config['db_res'],
            x_label=config['feature_x'], y_label=config['feature_y'],
            colors=config['db_colors'],
            xlim=config['db_xlim'], ylim=config['db_ylim'],
            vmin=config['db_vmin'], vmax=config['db_vmax'],
            figsize=config['db_figsize'],
            colorbar=True, ticks=None,
            annot={
                'text':utils.get_param_string(best_params).strip('{}')\
                    .replace('\'','').replace(',','\n').replace('\n ', '\n'),
                'x':config['db_annot_x'], 'y':config['db_annot_y']},
            scatter=True, subsample=0.01, label_cla=config['label_cla'],
            scatter_legend=False, colors_scatter=config['db_colors_scatter'],
            dist=True, nbins=config['db_nbins'],
            filename=output_path+"decision_boundary/overlay_db_best_model")


        # Plot decision boundary of the best model without scattered plot.
        plot_db.decision_boundary(
            model=model, df=df_train, features=features, h=config['db_res'],
            x_label=config['feature_x'], y_label=config['feature_y'],
            colors=config['db_colors'],
            xlim=config['db_xlim'], ylim=config['db_ylim'],
            vmin=config['db_vmin'], vmax=config['db_vmax'],
            figsize=config['db_figsize'],
            colorbar=True, ticks=None,
            annot={
                'text':utils.get_param_string(best_params).strip('{}')\
                    .replace('\'','').replace(',','\n').replace('\n ', '\n'),
                'x':config['db_annot_x'], 'y':config['db_annot_y']},
            scatter=False, subsample=0.01, label_cla=config['label_cla'],
            scatter_legend=False, colors_scatter=config['db_colors_scatter'],
            dist=True, nbins=config['db_nbins'],
            filename=output_path+"decision_boundary/db_best_model")


    #---------------------------------------------------------------------------
    # Examine residual
    #---------------------------------------------------------------------------
    if plot_residual:
        # Plot distribution of residual
        # Calculate residual
        res_train = pred_train[config['label_reg']] - pred_train['pred']
        res_test = pred_test[config['label_reg']] - pred_test['pred']
        
        # Plot distribution of prediction and target
        plot.plot_dist_hue(
            df= pd.concat(
                [res_train, res_test], keys=['Train', 'Test'], axis=1)\
                .stack()\
                .reset_index(level=-1)\
                .rename(
                    {0:'Residual (%s - Prediction)' %config['label_reg']},
                    axis=1),
            x='Residual (%s - Prediction)' %config['label_reg'], hue="level_1",
            hue_str={'Train':'Train (normalized)', 'Test':'Test (normalized)'},
            hist_type='step', ylabel='Samples', norm=True,
            n_bins=config['residual_n_bins'],
            figsize=(8,5), filename=output_path+'residual/res_dist')


        # Plot distribution of prediction and target
        plot.plot_dist_hue(
            df=pred_train[[config['label_reg'], 'pred']].stack()\
                .reset_index(level=-1).rename({0:config['label_reg']}, axis=1),
            x=config['label_reg'], hue="level_1",
            hue_str={config['label_reg']:'True', 'pred':'Prediction'},
            hist_type='step', ylabel='Samples', norm=False,
            n_bins=config['residual_n_bins'],
            figsize=(8,5), filename=output_path+'residual/dist_train')

        plot.plot_dist_hue(
            df=pred_test[[config['label_reg'], 'pred']].stack()\
                .reset_index(level=-1).rename({0:config['label_reg']}, axis=1),
            x=config['label_reg'], hue="level_1",
            hue_str={config['label_reg']:'True', 'pred':'Prediction'},
            hist_type='step', ylabel='Samples', norm=False,
            n_bins=config['residual_n_bins'],
            figsize=(8,5), filename=output_path+'residual/dist_test')

        # Plot prediction vs target
        plot.plot_reg(
            df=pred_train, x=config['label_reg'], y='pred',
            x_label=config['label_reg'], y_label='Prediction',
            figsize=(10, 10), filename=output_path+'residual/res_scatter_train',
            fit_reg=True, scatter_kws={'linewidth':1, 's':1, 'color':'dodgerblue'},
            line_kws={'color':'crimson'})

        plot.plot_reg(
            df=pred_test, x=config['label_reg'], y='pred',
            x_label=config['label_reg'], y_label='Prediction',
            figsize=(10, 10), filename=output_path+'residual/res_scatter_test',
            fit_reg=True, scatter_kws={'linewidth':1, 's':1, 'color':'slategray'},
            line_kws={'color':'crimson'})


            


    #---------------------------------------------------------------------------
    # Save output as csv
    #---------------------------------------------------------------------------
    if save_csv:
        utils.create_folder(output_path+'csv/summary.csv')
        # Save the summary
        if not read_last:
            cv_results.to_csv(output_path+'csv/cv_results.csv')
        # Save
        if run_backtest:
            try:
                df_backtest_train.to_csv(output_path+'csv/cum_return_train.csv')
                df_backtest_test.to_csv(output_path+'csv/cum_return_test.csv')
                df_diff_train.to_csv(output_path+'csv/backtest_diff_train.csv')
                df_diff_test.to_csv(output_path+'csv/backtest_diff_test.csv')
            except IOError:
                io.error("Cannot save backtesting results.")
        # Save ANOVA results
        try:
            for key in anova_results:
                if key in ['f_stats', 'p_values', 'best_params']:
                    pd.DataFrame.from_dict(anova_results[key], orient='index')\
                        .to_csv(output_path+'csv/anova_%s.csv' %key)
            # All tukey test results
            for metric in anova_results['tukey_all_results'].keys():
                if type(anova_results['tukey_all_results'][metric]) == pd.DataFrame:
                    anova_results['tukey_all_results'][metric].to_csv(
                            output_path+'csv/tukey_all_results_%s.csv' % metric)
            # Tukey test results related to the top performing model
            if type(anova_results['tukey_top_results']) == pd.DataFrame:
                anova_results['tukey_top_results'].to_csv(
                    output_path+'csv/tukey_top_results.csv')
            # Save ID of the selected model
            pd.DataFrame(
                data=[anova_results['id_selected_model']],
                columns=['id_selected_model']).to_csv(
                    output_path+'csv/id_selected_model.csv')
        except IOError:
            io.error("Cannot save ANOVA results.")
        # Save predictions
        if run_backtest or plot_decision_boundary:
            try:
                pred_train.to_csv(output_path+'csv/pred_train.csv')
                pred_test.to_csv(output_path+'csv/pred_test.csv')
            except IOError:
                io.error("Cannot save predictions.")
        # Save model evaluation
        if model_evaluation:
            pd.DataFrame([
                model_evaluation_train, model_evaluation_test],
                index=['Train', 'Test'])\
            .applymap(lambda x:x[0])\
            .to_csv(output_path+'csv/model_evaluation.csv')

    io.message("Successfully completed all tasks!")

    return {
        'cv_results': cv_results, 'anova_results': anova_results,
        'pred_train': pred_train, 'pred_test': pred_test,
        'model_evaluation_train': model_evaluation_train,
        'model_evaluation_test': model_evaluation_test,
        'cum_return_train': df_backtest_train, 'cum_return_test': df_backtest_test,
        'model': model}
            


