from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import uniform
from statsmodels.distributions.empirical_distribution import ECDF

from .evaluation import false_positive_rate, power

rcParams.update({"figure.autolayout": True})


def pvalues_hist(pvalues, bins=20, title=None, fname=None, figsize=(6, 4)):
    """
    Plot histogram of p-values.

    Args:
        pvalues (array-like): List of p-values.
        bins (int, optional): The number of bins. Defaults to 20.
        title (str, optional): Title of the figure. Defaults to None.
        fname (str, optional): File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (6, 4).
    """
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xlabel("p-value")
    plt.ylabel("frequency")
    plt.hist(pvalues, bins=bins, range=(0, 1))
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
    plt.clf()
    plt.close()


def pvalues_qqplot(pvalues, plot_pos=None, title=None, fname=None, figsize=(4, 4)):
    """
    Plot uniform Q-Q plot of p-values.

    Args:
        pvalues (array-like): List of p-values.
        plot_pos (array-like, optional): Plotting positions. If None, default plotting
            positions will be used. Defaults to None.
        title (str, optional): Title of the figure. Defaults to None.
        fname (str, optional): File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (4, 4).
    """
    n = len(pvalues)
    plot_pos = plot_pos or [k / (n + 1) for k in range(1, n + 1)]
    t_quantiles = list(map(uniform.ppf, plot_pos))  # theoretical
    e_quantiles = list(map(ECDF(pvalues), plot_pos))  # empirical
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xlabel("theoretical quantiles of Unif(0, 1)")
    plt.ylabel("empirical quantiles of p-values")
    plt.plot([0, 1], [0, 1])
    plt.plot(t_quantiles, e_quantiles, marker=".", linestyle="None")
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
    plt.clf()
    plt.close()


class SummaryFigure:
    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data = defaultdict(list)

    def add_value(self, value, label, xloc):
        self.data[label].append((xloc, value))

    def plot(self, fname=None, sort_xlocs=True):
        """
        Plot the figure.

        Args:
            fname (str, optional): File name. If `fname` is given, the plotted figure
                will be saved as a file. Defaults to None.
            sort_xlocs (bool, optional): If True, xlocs will be sorted in ascending
                order.
        """
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        plt.ylim(0, 1)
        for label, list_ in self.data.items():
            if sort_xlocs:
                list_.sort(key=lambda x: x[0])
            xlocs = [v[0] for v in list_]
            values = [v[1] for v in list_]
            plt.plot(xlocs, values, label=label)
        plt.legend()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, transparent=True)
        plt.clf()
        plt.close()


class FprFigure(SummaryFigure):
    """
    Plot a fpr summary figure

    Usage1:
        fig = FprFigure(title='Figure1', xlabel='parameter')
        fig.add_fpr(0.1, 'naive', 1)
        fig.add_fpr(0.3, 'naive', 2)
        fig.add_fpr(0.8, 'naive', 3)
        fig.add_fpr(0.05, 'selective', 1)
        fig.add_fpr(0.05, 'selective', 2)
        fig.add_fpr(0.05, 'selective', 3)
        fig.plot()

    Usage2:
        fig = FprFigure(title='Figure2', xlabel='setting')
        fig.add_pvalues([0.8, 0.01, 0.06], 'naive', 'hoge')
        fig.add_pvalues([0.03, 0.05, 0.2], 'naive', 'foo')
        fig.add_pvalues([0.01, 0.0, 0.02], 'naive', 'bar')
        fig.plot(file_name='figure2.pdf', sort_xlocs=False)
    """

    def __init__(self, title=None, xlabel=None, ylabel="false positive rate"):
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)

    def add_pvalues(self, pvalues, label, xloc, alpha=0.05):
        """
        Add p-values to the figure.

        Args:
            pvalues (array-like): List of p-values.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
            alpha (float, optional): Significance level.
        """
        fpr = false_positive_rate(pvalues, alpha=alpha)
        self.add_fpr(fpr, label, xloc)

    def add_fpr(self, fpr, label, xloc):
        """
        Add a fpr value to the figure.

        Args:
            fpr (float): FPR value.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
        """
        self.add_value(fpr, label, xloc)


class PowerFigure(SummaryFigure):
    """
    Plot a power summary figure

    Usage1:
        fig = PowerFigure(title='Figure1', xlabel='parameter')
        fig.add_power(0.95, 'naive', 1)
        fig.add_power(0.95, 'naive', 2)
        fig.add_power(0.95, 'naive', 3)
        fig.add_power(0.1, 'selective', 1)
        fig.add_power(0.3, 'selective', 2)
        fig.add_power(0.8, 'selective', 3)
        fig.plot()

    Usage2:
        fig = PowerFigure(title='Figure2', xlabel='setting')
        fig.add_pvalues([0.8, 0.01, 0.2], 'naive', 'hoge')
        fig.add_pvalues([0.8, 0.01, 0.03], 'naive', 'foo')
        fig.add_pvalues([0.02, 0.01, 0.03], 'naive', 'bar')
        fig.plot(file_name='figure2.pdf', sort_xlocs=False)
    """

    def __init__(self, title=None, xlabel=None, ylabel="power"):
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)

    def add_pvalues(self, pvalues, label, xloc, alpha=0.05):
        """
        Add p-values to the figure.

        Args:
            pvalues (array-like): List of p-values.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
            alpha (float, optional): Significance level.
        """
        power_ = power(pvalues, alpha=alpha)
        self.add_power(power_, label, xloc)

    def add_power(self, power, label, xloc):
        """
        Add a power value to the figure.

        Args:
            power (float): Power value.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
        """
        self.add_value(power, label, xloc)
