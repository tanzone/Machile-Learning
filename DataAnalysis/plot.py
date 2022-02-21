import random

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as ply
from datetime import datetime

from DataAnalysis.utility import *

# Costante di colori per i plot
COLORS = ["brown", "orange", "lawngreen", "yellow", "aqua", "blue", "pink", "violet", "purple"]
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


def plotSomething(df, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", save: bool = False):
    x = df[colX].tolist()
    y = df[colY].tolist()

    data = go.Scatter(x=x, y=y, name="", line=dict(color=COLORS[random.randint(0, len(COLORS) - 1)], width=4))
    # dash="dot" o dash="dashdot" da mettere nel dict

    layout = dict(title="Plot {} on: {} - {}".format(name, colX, colY), xaxis=dict(title=colX), yaxis=dict(title=colY),
                  plot_bgcolor="black")

    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")
    if not save:
        # DA FARE, cancellare il file temp
        pass


def plotStocksTrend(datasets, colX: str = "Date", colY: str = "CloseUSD", name: str = "stocksTrend"):
    color = COLORS
    data = list()
    for key in datasets:
        df = datasets[key]
        x = df[colX].tolist()
        y = df[colY].tolist()
        data.append(
            go.Scatter(x=x, y=y, name=key, line=dict(color=color.pop(random.randint(0, len(COLORS) - 1)), width=3)))

    layout = dict(title="Plot {} on: {} - {}".format(name, colX, colY), xaxis=dict(title=colX), yaxis=dict(title=colY),
                  plot_bgcolor="black")

    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")


def _controlFeatures(df, cols):
    if cols is None:
        cols = STANDARD_FEATURE
        multipleFeature(df, cols)
    return cols


def _plotFeatures(df, col, colX):
    dfSupp = df[df[col] == 1]
    x = dfSupp[colX].tolist()

    if col.split("-")[0] == "BUY":
        color = "green"
        y = dfSupp["BuyPrice"].tolist()
        return color, x, y
    elif col.split("-")[0] == "SELL":
        color = "red"
        y = dfSupp["SellPrice"].tolist()
        return color, x, y


def plotStockFeatures(df, colX: str = "Date", colY: str = "CloseUSD", cols=None, name: str = "StockFeature",
                      dateStart=DATE_START, dateEnd=DATE_END):
    # subset con il limite delle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    cols = _controlFeatures(df, cols)

    # plot della stock
    x = df[colX].tolist()
    y = df[colY].tolist()

    color = COLORS
    title = "Plot {} on: {} - {}".format(name, colX, colY)
    data = list()
    data.append(go.Scatter(x=x, y=y, name=name, line=dict(color=color.pop(random.randint(0, len(COLORS) - 1)), width=4)))

    # plot delle features
    for col in cols:
        if col.split("-")[0] == "BUY" or col.split("-")[0] == "SELL":
            color, x, y = _plotFeatures(df, col, colX)
            data.append(
                go.Scatter(x=x, y=y, name=col, mode="markers", marker=dict(color=color, size=8, line=dict(
                                                                                    color=color, width=3))))
            title += " - {}".format(col)
        else:
            y = df[col].tolist()
            data.append(
                go.Scatter(x=x, y=y, name=col, line=dict(color=color.pop(random.randint(0, len(COLORS) - 1)),
                                                         width=2, dash="dash")))
            title += " - {}".format(col)

    # show
    layout = dict(title=title, xaxis=dict(title=colX), yaxis=dict(title=colY), plot_bgcolor="black")
    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")
