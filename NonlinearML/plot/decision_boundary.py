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
    model, df, features, rank=False,
    h=0.01, x_label="", y_label="", xlim=False, ylim=False,
    title=False, title_loc='center', annot=False, vlines=[], hlines=[],
    scatter=False, subsample=0.01, scatter_legend=False,
    label_cla=None, dist=False, nbins=20,
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
        annot: Dictionary containing the following:
            {'x': x coordinateof annotation,
             'y': y coordinateof annotation,
             'text': text to display}
            If False, do not display any annotation. 
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
    if rank:
        z = model.predict(df_mesh, None)
    else:
        z = model.predict(df_mesh)
    z = z.reshape(xx.shape)
    # Put the result into a color plot
    cmap = ListedColormap(sns.color_palette(colors).as_hex())
    # Create figure
    if not dist:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    # Plot distribution of two features
    elif dist:
        # Create joint plot using seaborn
        g = sns.JointGrid(
            x=features[0], y=features[1], data=df,
            height=figsize[1], space=0.3, ratio=6)
        g = g.plot_marginals(
            sns.distplot, kde=False, bins=nbins,
            color="gray", hist_kws={
                "histtype": "stepfilled", "linewidth": 1.5,
                "alpha": 1, "color": "gray", "edgecolor": "black"})
        # Set axes and figure to joint plot
        ax = g.ax_joint
        fig = g.fig
    # Plot decision boundaries
    im = ax.pcolormesh(xx, yy, z, cmap=cmap)
    # Add scatter plot
    if scatter:
        if type(subsample)==float:
            df_sub = df.sample(frac=subsample)
        elif type(subsample)==pd.DataFrame:
            df_sub = subsample
        for i, cls in enumerate(sorted(df_sub[label_cla].unique())):
            _df = df_sub.loc[df_sub[label_cla]==cls]
            ax.scatter(
                x=_df[features[0]], y=_df[features[1]], s=50,
                edgecolors='black', c=colors[i], label=cls)
            if scatter_legend==True:
                ax.legend(loc='lower right')
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
    #plt.tight_layout()
    if ticks:
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








'''
    Develop a function to visualize decision boundary
        for any classification models in 2D
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def retrieve_n_class_color_cubic(N):
    """
    retrive color code for N given classes
    Input: class number
    Output: list of RGB color code
    """

    # manualy encode the top 8 colors
    # the order is intuitive to be used
    color_list = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 0),
        (1, 1, 1)
    ]

    # if N is larger than 8 iteratively generate more random colors
    np.random.seed(1)  # pre-define the seed for consistency

    interval = 0.5
    while len(color_list) < N:
        the_list = []
        iterator = np.arange(0, 1.0001, interval)
        for i in iterator:
            for j in iterator:
                for k in iterator:
                    if (i, j, k) not in color_list:
                        the_list.append((i, j, k))
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.0

    return color_list[:N]


def decision_boundary_ext(
    model, filename, colors, dim_red_method='pca',
    X=None, Y=None, xrg=None, yrg=None,
    Nx=300, Ny=300, scatter_sample=None,
    figsize=[6, 6], alpha=0.7,
    random_state=111):
    '''
    Plot decision boundary for any two dimension classification models
        in sklearn.
    Input:
        model: sklearn classification model class - already fitted
                (with "predict" and "predict_proba" method)
        dim_red_method: sklearn dimension reduction model
                (with "fit_transform" and "inverse_transform" method)
        xrg (list/tuple): xrange
        yrg (list/tuple): yrange
        Nx (int): x axis grid size
        Ny (int): y axis grid size
        X (nparray): dataset to project over decision boundary (X)
        Y (nparray): dataset to project over decision boundary (Y)
        figsize, alpha are parameters in matplotlib
    Output:
        matplotlib figure object
    '''

    # check model is legit to use
    try:
        getattr(model, 'predict')
    except:
        print("model do not have method predict 'predict' ")
        return None

    use_prob = True
    try:
        getattr(model, 'predict_proba')
    except:
        print("model do not have method predict 'predict_proba' ")
        use_prob = False

    # convert X into 2D data
    ss, dr_model = None, None
    if X is not None:
        if X.shape[1] == 2:
            X2D = X
        elif X.shape[1] > 2:
            # leverage PCA to dimension reduction to 2D if not already
            ss = StandardScaler()
            if dim_red_method == 'pca':
                dr_model = PCA(n_components=2)
            elif dim_red_method == 'kernal_pca':
                dr_model = KernelPCA(n_components=2,
                                     fit_inverse_transform=True)
            else:
                print('dim_red_method {0} is not supported'.format(
                    dim_red_method))

            X2D = dr_model.fit_transform(ss.fit_transform(X))
        else:
            print('X dimension is strange: {0}'.format(X.shape))
            return None

        # extract two dimension info.
        x1 = X2D[:, 0].min() - 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        x2 = X2D[:, 0].max() + 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        y1 = X2D[:, 1].min() - 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())
        y2 = X2D[:, 1].max() + 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())

    # inti xrg and yrg based on given value
    if xrg is None:
        if X is None:
            xrg = [-10, 10]
        else:
            xrg = [x1, x2]

    if yrg is None:
        if X is None:
            yrg = [-10, 10]
        else:
            yrg = [y1, y2]

    # generate grid, mesh, and X for model prediction
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)

    xx, yy = np.meshgrid(xgrid, ygrid)
    X_full_grid = np.array(list(zip(np.ravel(xx), np.ravel(yy))))

    # initialize figure & axes object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # get data from model predictions
    if dr_model is None:
        Yp = model.predict(X_full_grid)
        if use_prob:
            Ypp = model.predict_proba(X_full_grid)
        else:
            Ypp = pd.get_dummies(Yp).values
    else:
        X_full_grid_inverse = ss.inverse_transform(
            dr_model.inverse_transform(X_full_grid))

        Yp = model.predict(X_full_grid_inverse)
        if use_prob:
            Ypp = model.predict_proba(X_full_grid_inverse)
        else:
            Ypp = pd.get_dummies(Yp).values

    # retrieve n class from util function
    nclass = Ypp.shape[1]
    colors = np.array(retrieve_n_class_color_cubic(N=nclass))

    # get decision boundary line
    Yp = Yp.reshape(xx.shape)
    Yb = np.zeros(xx.shape)

    Yb[:-1, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[:-1, :])
    Yb[1:, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[1:, :])
    Yb[:, :-1] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, :-1])
    Yb[:, 1:] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, 1:])

    # plot decision boundary first
    ax.imshow(Yb, origin='lower', interpolation=None, cmap='Greys',
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=1.0)

    # plot probability surface
    zz = np.dot(Ypp, colors[:nclass, :])
    zz_r = zz.reshape(xx.shape[0], xx.shape[1], 3)
    ax.imshow(zz_r, origin='lower', interpolation=None,
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=alpha)

    # add scatter plot for X & Y if given
    if X is not None:
        # down sample point if needed
        if Y is not None:
            if scatter_sample is not None:
                X2DS, _, YS, _ = train_test_split(X2D, Y, stratify=Y,
                                                  train_size=scatter_sample,
                                                  random_state=random_state)
            else:
                X2DS = X2D
                YS = Y
        else:
            if scatter_sample is not None:
                X2DS, _ = train_test_split(X2D, train_size=scatter_sample,
                                           random_state=random_state)
            else:
                X2DS = X2D

        # convert Y into point color
        if Y is not None:
            # Map between unique value and color index
            value_to_color = {
                value:index for index, value in enumerate(np.unique(Y))}
            cYS = [colors[value_to_color[value]] for value in YS]

        if Y is not None:
            ax.scatter(X2DS[:, 0], X2DS[:, 1], c=cYS)
        else:
            ax.scatter(X2DS[:, 0], X2DS[:, 1])

    # add legend on each class
    colors_bar = []
    for v1 in colors[:nclass, :]:
        v1 = list(v1)
        v1.append(alpha)
        colors_bar.append(v1)

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors_bar[i],
                              label="Class {k}".format(k=i))
               for i in range(nclass)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),
               loc=2, borderaxespad=0., framealpha=0.5)

    # make the figure nicer
    ax.set_title('Classification decision boundary')
    if dr_model is None:
        ax.set_xlabel('Raw axis X')
        ax.set_ylabel('Raw axis Y')
    else:
        ax.set_xlabel('Dimension reduced axis 1')
        ax.set_ylabel('Dimension reduced axis 2')
    ax.set_xlim(xrg)
    ax.set_ylim(yrg)
    ax.set_xticks(np.arange(xrg[0], xrg[1], (xrg[1] - xrg[0])/5.))
    ax.set_yticks(np.arange(yrg[0], yrg[1], (yrg[1] - yrg[0])/5.))
    ax.grid(True)

    # Create output folder and save figure
    create_folder(filename)
    plt.savefig('%s.png' %filename)
    return











