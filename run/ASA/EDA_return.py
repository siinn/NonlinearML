#!/usr/bin/env python
# Import common python libraries
from datetime import datetime
import dateutil.relativedelta
import itertools
import matplotlib
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Import custom libraries
import NonlinearML.lib.cross_validation as cv
import NonlinearML.plot.backtest as plot_backtest
import NonlinearML.plot.plot as plot
import NonlinearML.lib.utils as utils

#-------------------------------------------------------------------------------
# Set user options
#-------------------------------------------------------------------------------
# Set input and output path
input_path = '/mnt/mainblob/nonlinearML/data/ASA/csv/ASA_G2_data.r5.p1.csv'

# Set input path for beta correlation data
beta_corr_path = '/mnt/mainblob/nonlinearML/data/ASA/csv/20200427_CORR.xls'
beta_corr_date = 'smDate'
beta_corr = ['PM6M', 'PM12M']

# Set available features and labels
features = ['Diff_Exp', 'PM_Exp']

# Feature used to create PM reversal flag
PM = 'PM_Exp'
thres = 0.03   # Threshold used to create momentum reversal flag

# Set return
FQ_RETURN = 'fqRelRet'

# Set train and test period
test_begin = "2011-01-01"
test_end = "2018-01-01"

# Set number of classes
n_classes = 5

# Set variable name for discretized features
feats_disc = ["%s_n%s" % (feat, n_classes) for feat in features]

# Set date column
time = 'smDate'

# return label
#month_return = "fqRelRet"

# colors map
cmap = matplotlib.cm.get_cmap('RdYlGn', 10)
colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
cmap=sns.color_palette(colors)




# Calculate standard error of mean or median
def getStandardError(x, confidence=1, median=False):
    """ Return standard error = confidence level * std / sqrt(N)"""
    scale = 1
    if median:
        # Scale factor for standard error of median
        scale=math.sqrt(math.pi/2)
    return scale*confidence*x.std() / (x.count()**(1/2))


#-------------------------------------------------------------------------------
# Create output folder
#-------------------------------------------------------------------------------

if __name__ == "__main__":

    # Set output path
    plot_path = 'output/ASA/EDA/return_by_time/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    #---------------------------------------------------------------------------
    # Read dataset
    #---------------------------------------------------------------------------
    # Read input csv
    df = pd.read_csv(input_path, parse_dates=[time])

    # Read beta correlation data
    df_beta = pd.read_excel(beta_corr_path)

    # Calculate relative forward month return
    FM_REL_RETURN = 'fmRelRet'
    FQ_REL_RETURN = 'fqRelRet'
    df[FM_REL_RETURN] = df.groupby('smDate')\
            .apply(
                lambda x:x['fmRet']- x['fmRet']\
                        .mean()).reset_index()['fmRet']
    if False :
        # Subtract monthly return
        df[FQ_REL_RETURN] = df.groupby('smDate')\
                .apply(
                    lambda x:x['fqRelRet']- x['fqRelRet']\
                            .mean()).reset_index()['fqRelRet']

    # Discretize feature
    df = utils.discretize_variables_by_month(
        df=df, variables=features,
        n_classes=n_classes, suffix="n%s" %n_classes,
        class_names=\
            ["Q%d" %x for x in range(n_classes,0,-1)],
        month=time)

    #---------------------------------------------------------------------------
    # Single factor return
    #---------------------------------------------------------------------------
    # Print statistics
    if False:

        for feat in features:
            # Calculate average by quintile
            time_periods = [
                ("1996-01-01","2017-11-30"),
                ("1996-01-01","2000-12-31"),
                ("2001-01-01","2010-12-31"),
                ("2011-01-01","2017-11-30"),
                    ]
            for ret in [FQ_REL_RETURN, FM_REL_RETURN]:
                print("Return = %s" %ret)
                for begin, end in time_periods:
                    df_subset = df.loc[(df[time]>=begin) & (df[time]<=end)]
                    df_top = df_subset.loc[df_subset["%s_n%s" % (feat, n_classes)]=='1 (High)'].mean()
                    df_bot = df_subset.loc[df_subset["%s_n%s" % (feat, n_classes)]=='5 (Low)'].mean()
                    print("    > Begin:%s, end:%s, Q1 mean=%.3f, Q5 mean=%.3f" % (begin, end, df_top[ret], df_bot[ret]))
        
                for begin, end in time_periods:
                    df_subset = df.loc[(df[time]>=begin) & (df[time]<=end)]
                    df_top = df_subset.loc[df_subset["%s_n%s" % (feat, n_classes)]=='1 (High)'].median()
                    df_bot = df_subset.loc[df_subset["%s_n%s" % (feat, n_classes)]=='5 (Low)'].median()
                    print("    > Begin:%s, end:%s, Q1 median=%.3f, Q5 median=%.3f" % (begin, end, df_top[ret], df_bot[ret]))
            # Calculate average by quintile
            #for ret in [FQ_REL_RETURN, FM_REL_RETURN]:
            for ret in [FQ_REL_RETURN]:
                df_mean = df.groupby([time, "%s_n%s" % (feat, n_classes)])[ret]\
                            .mean()\
                            .unstack()
    
                plot.plot_line_multiple_cols(
                    df=df_mean, x='index', list_y=["1 (High)", "%s (Low)" %n_classes],
                    legends=["1 (High)", "%s (Low)" %n_classes],
                    x_label="Time", y_label=feat, figsize=(20,6),
                    filename=plot_path+feat, legend_box=(0,-0.2),
                    marker='.', ms=20)

    #---------------------------------------------------------------------------
    # Two-factor return
    #---------------------------------------------------------------------------
    if True:

        for window in [6]:
            print("Creating plots for window size of %s" % window)

            # Group by two variables and time
            return_group = "%s-%s" %(features[0],features[1])
            df[return_group] = \
                df[feats_disc[0]].astype(str)\
                    .apply(lambda x:"%s %s" %(features[0][:-4], x))\
                + df[feats_disc[1]].astype(str)\
                    .apply(lambda x:"; %s %s" %(features[1][:-4], x))

            # Plot relative return of each group
            df_return = df.groupby([time, return_group])[FQ_REL_RETURN]\
                            .mean().unstack()\
                            .rolling(window=window).mean()

            # Calculate standard error of rolling mean
            df_err = df.groupby([time, return_group])[FQ_REL_RETURN]\
                    .apply(getStandardError).unstack()
                    #.rolling(window=window)\
                    #.apply(lambda x:np.sqrt(np.mean(np.array(x)*np.array(x))))

            #-------------------------------------------------------------------
            # Line plots in grid
            #-------------------------------------------------------------------
            plot.plot_line(
                df=df_return, x='index', columns=df_return.columns,
                n_rows=5, n_columns=5, ylog=[], ylim=[],
                title=[], x_label=[], y_label=[], grid='both',
                figsize=(25,25), filename=plot_path+"/window_%s/return_line" % window,
                xticks_minor=True, legends=df_return.columns,
                marker='o', color='dodgerblue')

            #-------------------------------------------------------------------
            # Line plots on the same plot
            #-------------------------------------------------------------------
            #GROUP_INTEREST = ['Diff Q1; PM Q1', 'Diff Q1; PM Q5', 'Diff Q5; PM Q1', 'Diff Q5; PM Q5']
            GROUP_INTEREST = ['Diff Q1; PM Q1', 'Diff Q5; PM Q5']

            # Merge return and its corresponding standard error
            df_return_and_error = pd.merge(
                df_return[GROUP_INTEREST].stack().reset_index().rename({0:FQ_REL_RETURN}, axis=1),
                df_err[GROUP_INTEREST].stack().reset_index().rename({0:FQ_REL_RETURN+"(err)"}, axis=1),
                left_on=[time, return_group], right_on=[time, return_group])

            # Prepare beta correlation data
            df_beta = df_beta.set_index("smDate")[['PM6M', 'PM12M']]\
                .stack().reset_index()\
                .rename({'level_1':return_group, 0:FQ_REL_RETURN},axis=1)
            # Add dummy uncertainty term
            df_beta[FQ_REL_RETURN+"(err)"] = 0

            # Beta correlation and mean return of PM/DIFF quintile
            plot.plot_line_groupby(
                df=pd.concat([df_return_and_error, df_beta]),
                x=time, y=FQ_REL_RETURN, groupby=return_group,
                yerr=FQ_REL_RETURN+"(err)",
                group_label={x:x for x in GROUP_INTEREST+['PM6M', 'PM12M']},
                #list_colors=[
                #    'limegreen',
                #    'dodgerblue',
                #    'orange',
                #    'crimson',
                #    'black', 'gray'],
                list_colors=[
                    'limegreen',
                    'crimson',
                    'dodgerblue', 'orange'],
                x_label="Time", y_label=FQ_REL_RETURN, linewidth=1,
                figsize=(20,6), filename=plot_path+"/window_%s/return_line_interest" %window,
                grid='both', marker='o', markersize=3)
            
            # Calculate binary reversal flag
            df_return_and_error = df_return_and_error.rename({'Diff_Exp-PM_Exp':'cat'},axis=1)
            df_reversal = pd.DataFrame(
                    df_return['Diff Q5; PM Q5'] - df_return['Diff Q1; PM Q1'] > thres)\
                .rename({0:'PM_Reversal'}, axis=1)\
                .astype(int)

            df_reversal = df_reversal.rename({'PM_Reversal':FQ_REL_RETURN}, axis=1)
            df_reversal['cat'] = 'PM Reversal'
            df_reversal[FQ_REL_RETURN+"(err)"] = 0
            df_reversal = df_reversal.reset_index()

            # Return of PM/DIFF quintile and PM reversal flag
            plot.plot_line_groupby(
                df=pd.concat([df_reversal, df_return_and_error]),
                x=time, y=FQ_REL_RETURN, groupby='cat',
                yerr=FQ_REL_RETURN+"(err)",
                group_label={x:x for x in GROUP_INTEREST + ['PM Reversal']},
                list_colors=[
                    'limegreen', 'crimson', 'orange'],
                x_label="Time", y_label=FQ_REL_RETURN, linewidth=1,
                figsize=(20,6), filename=plot_path+"/window_%s/pm_reversal" %window,
                grid='both', marker='o', markersize=3)
            

    print("Successfully completed all tasks")


