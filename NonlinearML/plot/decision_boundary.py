from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from NonlinearML.lib.utils import create_folder
from NonlinearML.lib.utils import get_param_string
from NonlinearML.plot.plot import *
from NonlinearML.plot.style import load_matplotlib
plt = load_matplotlib()


#-------------------------------------------------------------------------------
# Decision boundary plots
#-------------------------------------------------------------------------------
#def get_param_string(params):
#    """ Get name as a string."""
#    names = []
#    for key in params:
#        if 'name' in dir(params[key]):
#            names.append(key+'='+params[key].name)
#        else:
#            names.append(key+'='+str(params[key]))
#        """ Todo: write a function to extract layer info
#            if 'layers' in dir(params[key]):"""
#    return ",".join(names)

def decision_boundary(
    model, df, features, h=0.01, x_label="", y_label="", xlim=False, ylim=False,
    title=False, title_loc='center', annot=False, vlines = [], hlines = [],
    colors=["#BA2832", "#F3F2F2", "#2A71B2"], figsize=(8,4), ticks=[],
    filename=""):
    ''' Plot decision boundary of trained model.
    Args:
        model: Fitted model that has .predict method
        df: input dataframe containing two features. Used to extract feature
            domain (mix, max values). Column order decides x and y. First
            feature becomes x, second feature becomes y. 
        features: list of features. ex. ["AG", "FCFA"]
        h: step size when creating grid
        vlines, hlines: vertical and horizontal lines to draw given in list
            Example: vlines=[-0.5, 0, 0.5]
        others: plotting option
    Returns:
        None
    '''
    print("Plotting decision boundary with filename: \n >%s" %filename)
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
    fig, ax = plt.subplots(1,1, figsize=figsize)
    im = ax.pcolormesh(xx, yy, z, cmap=cmap)
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, loc=title_loc)
    if annot:
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


def decision_boundary_multiple_hparmas(param_grid, label, db_annot_x, db_annot_y, **kwargs):
    ''' Plot decision boundary for each hyperparameter set. This is a wrapper
        of 'decision_boundary' function.
    Args:
        param_grid: Hyperparamater grid to search.
        label: classification label
        db_annot_x, db_annot_y: Location of annotation that displays parameters.
        **kwargs:
            Arguments for 'decision_boundary'. Only difference is
            that df must be training data.
            In 'decision_boundary', this could be training or test data
            as it is only used to extract range of the plot.
    Returns:
        None
    '''
    print('\nCreating decision boundary plots for each combination of',
          ' hyperparameters')
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
        print(' > Experiment (%s/%s)' % (count, n_experiments))
        print(' > Parameters: %s' % str(params))
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
        
    print("\nPlotting decision boundary plot using partial dependence plot."\
           + " Filename: \n >%s" %filename)
    # Create figures for partial dependence plot using Skater
    interpreter = Interpretation(
        examples, feature_names=list(examples.columns))
    im_model = InMemoryModel(
        model.predict_proba, examples=examples, target_names=target_names)
    df_pd = interpreter.partial_dependence.partial_dependence(
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


