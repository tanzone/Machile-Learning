import random

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as ply

from DataAnalysis.utility import *

# Costante di colori per i plot
COLOR_LINE = "purple"
COLORS = []
for i in range(30):
    COLORS.append('#%06X' % random.randint(0, 0xFFFFFF))
# COLORS = ["brown", "orange", "lawngreen", "yellow", "aqua", "blue", "pink", "violet", "purple"]
DATE_START = "1500-01-01"
DATE_END = "3000-01-01"
STANDARD_FEATURE = ["SMA-5", "SMA-100", "SMA-200", "BUY-200-10", "SELL-5"]


def plotPie(*datasets, col: str = "Index"):
    for df in datasets:
        plt.figure(figsize=(10, 10))
        df[col].value_counts().plot.pie(autopct="%1.1f%%")
        plt.show()


# DA FARE, plot bar dei valori presenti in ogni stocks
def plotBar(*datasets, col: str = "Index"):
    pass


def _plotly(title, name, colX, colY, data):
    layout = dict(title=title, xaxis=dict(title=colX), yaxis=dict(title=colY), plot_bgcolor="black")
    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")


def plotSomething_line(df, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", color=[COLOR_LINE],
                       loop: bool = False):
    x = df[colX].tolist()
    y = df[colY].tolist()

    data = go.Scatter(x=x, y=y, name=name, line=dict(color=color.pop(random.randint(0, len(COLORS) - 1)), width=4))

    if not loop:
        # show
        title = "Plot {} on: {} - {}".format(name, colX, colY)
        _plotly(title, name, colX, colY, data)
    else:
        return data


def plotSomething_dash(df, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", color=[COLOR_LINE],
                       loop: bool = False):
    x = df[colX].tolist()
    y = df[colY].tolist()

    data = go.Scatter(x=x, y=y, name=colY, line=dict(color=color.pop(random.randint(0, len(COLORS) - 1)),
                                                     width=2, dash="dash"))

    if not loop:
        # show
        title = "Plot {} on: {} - {}".format(name, colX, colY)
        _plotly(title, name, colX, colY, data)
    else:
        return data


def plotSomething_marker(df, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", color=[COLOR_LINE],
                         loop: bool = False):
    x = df[colX].tolist()
    y = df[colY].tolist()

    data = go.Scatter(x=x, y=y, name=colY, mode="markers", marker=dict(color=color, size=8, line=dict(
        color=color, width=3)))

    if not loop:
        # show
        title = "Plot {} on: {} - {}".format(name, colX, colY)
        _plotly(title, name, colX, colY, data)
    else:
        return data


# Da sistemare
def plotStocksTrend(datasets, colX: str = "Date", colY: str = "CloseUSD", name: str = "stocksTrend"):
    data = list()
    for key in datasets:
        df = datasets[key]
        data.append(plotSomething_line(df, colX, colY, key, COLORS, True))

    # show
    title = "Plot {} on: {} - {}".format(name, colX, colY)
    _plotly(title, name, colX, colY, data)


def _controlFeatures(df, cols):
    if cols is None:
        cols = STANDARD_FEATURE
        multipleFeature(df, cols)
    return cols


def _setFeatures(df, col):
    if col.split("-")[0] == "BUY":
        return df[df[col] == 1], "green", "BuyPrice"
    elif col.split("-")[0] == "SELL":
        return df[df[col] == 1], "red", "SellPrice"


def plotStockFeatures(df, colX: str = "Date", colY: str = "CloseUSD", cols=None, name: str = "StockFeature",
                      dateStart=DATE_START, dateEnd=DATE_END):
    # subset con il limite delle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    cols = _controlFeatures(df, cols)

    # plot della stock
    title = "Plot {} on: {} - {}".format(name, colX, colY)

    data = list()
    data.append(plotSomething_line(df, colX, colY, name, [COLOR_LINE], True))

    # plot delle features
    for col in cols:
        if col.split("-")[0] == "BUY" or col.split("-")[0] == "SELL":
            dfPlot, colorPlot, colPlot = _setFeatures(df, col)
            data.append(plotSomething_marker(dfPlot, colX, colPlot, name, [colorPlot], True))
        else:
            data.append(plotSomething_dash(df, colX, colY, name, COLORS, True))

        title += " - {}".format(col)

    # show
    _plotly(title, name, colX, colY, data)


# Da sistemare
def plotStocksVolume(datasets, name: str = "temp"):
    data = list()
    for key in datasets:
        df = datasets[key]
        data.append(plotSomething_line(df, "Date", "Volume", key, COLORS, True))

    # show
    title = "Plot {} on: {} - {}".format(name, "Date", "Volume")
    _plotly(title, name, "Date", "Volume", data)
