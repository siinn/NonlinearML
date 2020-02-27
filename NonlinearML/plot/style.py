from cycler import cycler
import itertools
import matplotlib as mpl;mpl.use('agg') # use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def load_matplotlib():
    """ returns matplotlib with custom styles."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Main theme
    plt.style.use('fivethirtyeight')
    # Change default color cycle
    plt.rcParams['axes.prop_cycle']=cycler(color=sns.color_palette("hls", 20))
    # Change figure background
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25
    #plt.rcParams['lines.markeredgewidth'] = 1
    plt.rcParams["axes.grid"]  = False
    # Format axis
    plt.rcParams["date.autoformatter.month"] = "%Y-%m"
    plt.rcParams["date.autoformatter.year"] = "%Y-%m"

    return plt

def load_seaborn():
    """ return seaborn with custom styles."""
    return sns


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
