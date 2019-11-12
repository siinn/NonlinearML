from matplotlib.colors import ListedColormap
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
import pandas as pd
import itertools

from NonlinearML.lib.utils import create_folder
from NonlinearML.lib.utils import get_param_string
from NonlinearML.plot.plot import *
from NonlinearML.plot.style import load_matplotlib
from NonlinearML.plot.style import load_seaborn
import NonlinearML.lib.io as io
plt = load_matplotlib()
sns = load_seaborn()


#-------------------------------------------------------------------------------
# Decision boundary plots
#-------------------------------------------------------------------------------
def setup_axes(fig, rect, rotation, axisScale, axisLimits, doShift):
    """ Setup matplotlib axes"""
    tr_rot = Affine2D().scale(axisScale[0], axisScale[1]).rotate_deg(rotation)
    # This seems to do nothing
    if doShift:
        tr_trn = Affine2D().translate(-90,-5)
    else:
        tr_trn = Affine2D().translate(0,0)
    tr = tr_rot + tr_trn
    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=axisLimits)
    ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax)
    aux_ax = ax.get_aux_axes(tr)
    return ax, aux_ax

def decision_boundary(
    model, df, features, colors, rank=False,
    h=0.01, x_label="", y_label="", xlim=False, ylim=False, vmin=None, vmax=None,
    title=False, title_loc='center', annot=False, vlines=[], hlines=[],
    scatter=False, subsample=0.01, scatter_legend=False,
    label_cla=None, dist=False, nbins=20,
    colors_scatter=None, figsize=(8,4), colorbar=False, ticks=None,
    filename=""):
    ''' Plot decision boundary of trained model.
    Args:
        model: Fitted model that has .predict method
        df: input dataframe containing two features. Used to extract feature
            domain (mix, max values). Column order decides x and y. First
            feature becomes x, second feature becomes y. 
        features: list of features. ex. ["AG", "FCFA"]
        h: step size when creating grid
        annot: Dictionary containing the following:
            {'x': x coordinateof annotation,
             'y': y coordinateof annotation,
             'text': text to display}
            If False, do not display any annotation. 
        vmin, vmax: min and max value used for colorbar
        vlines, hlines: vertical and horizontal lines to draw given in list
            Example: vlines=[-0.5, 0, 0.5]
        scatter: Plot data as scatter plot on top of decision boundaries.
        subsample: subsampling rate used for scatter plot.
            ex. subsampling=0.1 means only 10% of data will be plotted.
            Used for scatter plot.
            
            If a dataframe is given, scatter plot is created using all samples
            in the dataframe.
        dist: Plot distribution of two features.
        n_bins: Number of bins in histogram.
        rank: If True, prediction is made by ranking the regression output.
        others: plotting option
    Returns:
        None
    '''
    io.message("Plotting decision boundary with filename:")
    io.message("\t" + filename)
    # Get x and y domain
    if xlim:
        x_min, x_max = xlim
    else:
        x_min, x_max = (df[features].iloc[:,0].min(),
                        df[features].iloc[:,0].max())
    if ylim:
        y_min, y_max = ylim
    else:
        y_min, y_max = (df[features].iloc[:,1].min(),
                        df[features].iloc[:,1].max())
    # Create grid of (x,y) pairs.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    df_mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])\
                .rename({i:features[i] for i in range(len(features))}, axis=1)

    # Make prediction for each point on grid
    z = model.predict(df_mesh)
    z = z.reshape(xx.shape)
    # Put the result into a color plot
    cmap = ListedColormap(sns.color_palette(colors).as_hex())
    if colors_scatter == None:
        colors_scatter = colors
    # Create figure
    if not dist:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    # Plot distribution of two features
    elif dist:
        # Create joint plot using seaborn
        g = sns.JointGrid(
            x=features[0], y=features[1], data=df,
            height=figsize[1], space=0.3, ratio=6)
        g.fig.set_figwidth(figsize[0])
        g.fig.set_figheight(figsize[1])
        g = g.plot_marginals(
            sns.distplot, kde=False, bins=nbins,
            color="gray", hist_kws={
                "histtype": "stepfilled", "linewidth": 1.5,
                "alpha": 1, "color": "gray", "edgecolor": "black"})
        # Set axes and figure to joint plot
        ax = g.ax_joint
        fig = g.fig
    # Plot decision boundaries
    im = ax.pcolormesh(xx, yy, z, cmap=cmap, vmin=vmin, vmax=vmax)
    # Add scatter plot
    if scatter:
        if type(subsample)==float:
            df_sub = df.sample(frac=subsample)
        elif type(subsample)==pd.DataFrame:
            df_sub = subsample
        for i, cls in enumerate(sorted(df_sub[label_cla].unique())):
            _df = df_sub.loc[df_sub[label_cla]==cls]
            ax.scatter(
                x=_df[features[0]], y=_df[features[1]], s=30,
                edgecolors='black', c=colors_scatter[i], label=cls)
            if scatter_legend==True:
                ax.legend(loc='lower right')
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, loc=title_loc)
    if annot:
        # Special treat for linear model
        if 'Ridge' in str(type(model)):
            for i, coef in enumerate(model.coef_):
                annot['text'] = annot['text']+'\ncoefficient %s=%.4f' % (i, coef)
            annot['text'] = annot['text']+'\nintercept=%.4f' % model.intercept_
        ax.text(
            annot['x'], annot['y'], annot['text'],
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(xx.min(), xx.max())
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(yy.min(), yy.max())
    if colorbar:
        colorbar = fig.colorbar(im, ax=ax)
        colorbar.set_ticks(ticks)
    # Draw vertical and horizontal lines.
    if vlines:
        for x in vlines:
            plt.axvline(x, linewidth=1, linestyle='--', color='black')
    if hlines:
        for y in hlines:
            plt.axhline(y, linewidth=1, linestyle='--', color='black')
    # Create output folder and save figure
    create_folder(filename)
    plt.savefig('%s.png' %filename)




def decision_boundary_multiple_hparmas(param_grid, label, db_annot_x, db_annot_y, rank=False, **kwargs):
    ''' Plot decision boundary for each hyperparameter set. This is a wrapper
        of 'decision_boundary' function.
    Args:
        param_grid: Hyperparamater grid to search.
        label: classification label
        db_annot_x, db_annot_y: Location of annotation that displays parameters.
        rank: If True, prediction is made by ranking the regression output.
        **kwargs:
            Arguments for 'decision_boundary'. Only difference is
            that df must be training data.
            In 'decision_boundary', this could be training or test data
            as it is only used to extract range of the plot.
    Returns:
        None
    '''
    io.title('Creating decision boundary plots of all combinations of hyperparameters')
    # Extract model from kwargs
    model = kwargs['model']
    df_train = kwargs['df']
    features = kwargs['features']
    filename = kwargs['filename']
    # Get all possible combination of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Loop over different values of k
    n_experiments = len(experiments)
    count=0
    for i, params in enumerate(experiments):
        count = count + 1
        io.message(' > Experiment (%s/%s)' % (count, n_experiments))
        io.message(' > Parameters:')
        io.message(["\t - "+x+"="+str(params[x]) for x in params])
        # Train model with given hyperparameter set
        model=model.set_params(**params)
        model.fit(df_train[features], df_train[label])
        # Assign trained model back to kwargs
        kwargs['model'] = model
        # Add counter to filename
        kwargs['filename'] = filename + "_%s" % i
        # Plot decision boundary of trained model
        decision_boundary(
            annot={
                'text':get_param_string(params).strip('{}')\
                    .replace('\'','').replace(',','\n').replace('\n ', '\n'),
                'x':db_annot_x, 'y':db_annot_y},
            rank=rank,
            **kwargs)
    return




def decision_boundary_pdp(
    model, examples, target_names, feature_interest, feature_other,
    grid_resolution=20, with_variance=True, figsize=(8,5),
    colors=["#3DC66D", "#F3F2F2", "#DF4A3A"],
    filename="plots/decesion_boundary_pdp", x_label="", y_label="",
    xlim=False, ylim=False, vlines = [], hlines = [], ticks=[]):
    ''' Create decision boundaries of two factors using partial dependence plot
    generated by Skater library. Details can be found below.
        https://github.com/oracle/Skater/
    Args:
        model: pretrained sklearn model. Or any model that has
            .predict_prob method.
        examples: dataframe containing features.
        target_names: name of classes. ex. ['T1', 'T2', 'T3'].
        feature_interest: feature of interest.
        feature_other: another features to compare with the feature of interest.
        grid_resolution: resolution of partial dependence plot.
        others: plotting options.
    Returns:
        None
    '''
    def _get_z(df, x, y, feature_interest, feature_other):
        '''Find row and return prediction given x and y.'''
        # Change data type to float
        return np.array(
            df.loc[
                (df[feature_interest]==x) & \
                (df[feature_other]==y)]['pred'])[0]
        
    io.message("\nPlotting decision boundary plot using partial dependence plot."\
           + " Filename: \n >%s" %filename)
    # Create figures for partial dependence plot using Skater
    interpretr = Interpretation(
        examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(
        model.predict_proba, examples=examples, target_names=target_names)
    df_pd = interpretr.partial_dependence.partial_dependence(
        # Pass feature_ids as list of tuples. ex. [('AG', 'FCFA')]
        feature_ids=[feature_interest, feature_other], 
        modelinstance=im_model, 
        grid_resolution=grid_resolution, sample=False,
        grid_range=(0,1), progressbar=False)

    # Retrieve grid of (x,y) pairs.
    xx, yy = np.meshgrid(
        df_pd[feature_interest].unique(), df_pd[feature_other].unique())
    # Make prediction by selecting class with the highest probability
    df_pd['pred'] = df_pd[target_names].idxmax(axis=1)
    # Get z coordinate of decision boundary by searching for (x,y)
    # from the dataframe.
    z = np.array(
        [
            [_get_z(df_pd,x,y, feature_interest, feature_other) for x in xx[0]]
            for y in yy[:,0]])
    # Put the result into a color plot
    cmap = ListedColormap(sns.color_palette(colors).as_hex())
    fig, ax = plt.subplots(1,1, figsize=figsize)
    im = ax.pcolormesh(xx, yy, z, cmap=cmap)
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(xx.min(), xx.max())
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(yy.min(), yy.max())
    plt.tight_layout()
    colorbar = fig.colorbar(im, ax=ax)
    if ticks:
        colorbar.set_ticks(ticks)
    # Draw vertical and horizontal lines.
    if vlines:
        for x in vlines:
            plt.axvline(x, linewidth=1, linestyle='--', color='black')
    if hlines:
        for y in hlines:
            plt.axhline(y, linewidth=1, linestyle='--', color='black')
    # Create output folder and save figure
    create_folder(filename)
    plt.savefig('%s.png' %filename)

