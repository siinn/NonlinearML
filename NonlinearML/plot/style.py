import itertools
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt

def load_matplotlib():
    """ returns matplotlib with custom styles."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('fivethirtyeight')
    #plt.style.use('tableau-colorblind10')
    #plt.style.use('seaborn-dark')
    #plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'
    plt.rcParams.update({'figure.max_open_warning': 0})
    return plt

def markers():
    """ returns marker cycler.
    Example: linestyle=next(line_cycler), hatch=next(hatch_cycler) """
    # Set cyler
    markers=('x', 'p', "|", '*', '^', 'v', '<', '>')
    marker_cyler = itertools.cycle(markers)
    return  marker_cycler

def lines():
    """ returns line cycler.
    Example: linestyle=next(line_cycler), hatch=next(hatch_cycler) """
    # Set cyler
    lines=("-","--","-.",":")
    line_cycler = itertools.cycle(lines)
    return line_cycler

def hatches():
    """ returns hatch cycler.
    Example: linestyle=next(line_cycler), hatch=next(hatch_cycler) """
    # Set cyler
    ''' Example: linestyle=next(line_cycler), hatch=next(hatch_cycler) '''
    hatch_cycler = itertools.cycle(hatchs)
    return hatch_cycler
