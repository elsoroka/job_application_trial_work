import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __():
    # this notebook can be run either as a standalone python file
    # or using marimo
    # `pip install marimo`
    # then `marimo run notebook.py`

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    return mo, mpl, np, pd, plt


@app.cell
def __(pd):
    data = pd.read_csv("merged_data.csv",
                       converters={'ts_event':pd.to_datetime,
                                   'bid_fill'      : float,
                                   'ask_fill'      : float,
                                   'Signed Volume' : float,
                                   'price'         : float,
                                   'best_bid'      : float,
                                   'best_ask'      : float,
                                   'mid_price'     : float,
                                  }
                      )
    # the data is not too big to read in at once, but if it was we would use chunksize=something to do it in bits

    data.dropna(inplace=True)
    return (data,)


@app.cell
def __(data, pd):
    # Binning the data into 10 second batches because this is done in the Handbook of Price Impact Modeling
    # This also allows us to use the efficient Pandas exponentially weighted moving average,
    # which only works for evenly spaced timeseries data.
    binned_data = data.groupby(pd.Grouper(key='ts_event', freq='10s'), dropna=True).agg(
                    {'bid_fill'      : "sum",
                     'ask_fill'      : "sum",
                     'Signed Volume' : "sum",
                     'price'         : "mean",
                     'best_bid'      : "min",
                     'best_ask'      : "max",
                     'mid_price'     : "mean",
                    }
                )
    return (binned_data,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Construct and code the linear OW model and nonlinear AFS model, and visualize the distri-
        bution of price impact based on the given data.

        Assumptions:

        * we use the parameters given in the paper in section 6.1
        * since the dataset we are given contains all the trades, we are going to use the price impact of all the trades (this is done in the Handbook of Price Impact Modeling example)
        * the dQ in the paper corresponds to the change in position, but in the handbook they use the signed volume so we do that here
        """
    )
    return


@app.cell
def __(binned_data, np):
    beta = 2.0
    lm = 0.0035
    adv = binned_data

    ewma = binned_data['Signed Volume'].ewm(halflife=beta).mean()

    linear_impact = lm*ewma

    afs_impact = lm*np.sign(ewma)*np.abs(ewma)**0.5
    return adv, afs_impact, beta, ewma, linear_impact, lm


@app.cell
def __(afs_impact, linear_impact, mpl, plt):
    formatter = mpl.dates.DateFormatter('%H:%M')
    figure, (axes1, axes2) = plt.subplots(2, sharex=True)
    axes1.xaxis.set_major_formatter(formatter)
    axes2.xaxis.set_major_formatter(formatter)

    axes1.plot(linear_impact)
    axes1.set_title("Linear impact model over time")
    axes2.plot(afs_impact)
    axes2.set_title("AFS impact model over time")

    plt.tight_layout()
    plt.savefig("task1.pdf")
    plt.show()
    return axes1, axes2, figure, formatter


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Some thoughts on this

        * The time looks off, presumably because it is in UTC. However, it was provided in UTC so I am not going to change it.
        * After-hours trades were left in.
        * The AFS model yields a smaller price impact for the same $\beta$ and $\lambda$ parameters, and smooths out the impact of large spikes in signed volume. This makes sense since AFS is a concave model, thus it penalizes larger trades less per share than mid-sized trades.
        """
    )
    return


if __name__ == "__main__":
    app.run()
