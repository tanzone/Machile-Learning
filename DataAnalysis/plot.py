import random
from DataAnalysis.utility import *

import plotly.graph_objs as go
import plotly.offline as ply
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

# Costante di colori per i plot
COLOR_LINE = "purple"
COLORS = []
for i in range(30):
    COLORS.append('#%06X' % random.randint(0, 0xFFFFFF))
# COLORS = ["brown", "orange", "lawngreen", "yellow", "aqua", "blue", "pink", "violet", "purple"]

DATE_START = "1500-01-01"
DATE_END = "3000-01-01"

STANDARD_FEATURE = ["SMA-5", "SMA-100", "SMA-200", "BUY-200-10", "SELL-5"]
STANDARD_FEATURE_MATLIB = ["CloseUSD", "SMA-5", "SMA-50"]


def plotPie(*datasets, col: str = "Index"):
    for df in datasets:
        plt.figure(figsize=(10, 10))
        df[col].value_counts().plot.pie(autopct="%1.1f%%")
        plt.show()


# DA FARE, plot bar dei valori presenti in ogni stocks
def plotBar(*datasets, col: str = "Index"):
    pass


def plotInfoStock(df, title: str = "temp"):
    df.plot(subplots=True, figsize=(10, 12))
    plt.title(title)
    plt.show()


def plotStocksTrend_matlib(datasets, cols=None):
    feature = True
    if cols is None:
        cols = ["CloseUSD"]
        feature = False
    i = 1
    for key in datasets:
        # Ne stampo 4 e basta per provare solo visivamente, non sono interessato ad una visualizzazione totale
        if i == 5:
            continue
        df = datasets[key]
        if feature:
            addFeatures(df, cols)
        df = df.loc[(df["Date"] > "2020-01-01") & (df["Date"] <= "2021-01-01")]
        plt.subplot(2, 2, i)
        for col in cols:
            plt.plot(df["Date"], df[col])
        plt.legend(cols)
        plt.ylabel("CloseUSD")
        plt.xlabel(None)
        plt.title(f"Closing Price of {key}")
        i += 1

    plt.show()


def plotStocksReturn_matlib(datasets):
    i = 1
    for key in datasets:
        # Ne stampo 4 e basta per provare solo visivamente, non sono interessato ad una visualizzazione totale
        if i == 5:
            continue
        df = datasets[key]
        stockReturn(df)
        df = df.loc[(df["Date"] > "2020-01-01") & (df["Date"] <= "2021-01-01")]
        plt.subplot(2, 2, i)
        plt.plot(df["Date"], df["RETURN"], marker='o')
        plt.ylabel("Return")
        plt.xlabel(None)
        plt.title(f"Daily return of {key}")
        i += 1

    plt.show()

def plotStocksReturn_matlib_bar(datasets):
    i = 1
    for key in datasets:
        # Ne stampo 4 e basta per provare solo visivamente, non sono interessato ad una visualizzazione totale
        if i == 5:
            continue
        df = datasets[key]
        stockReturn(df)
        df = df.loc[(df["Date"] > "2020-01-01") & (df["Date"] <= "2021-01-01")]
        plt.subplot(2, 2, i)
        df["RETURN"].hist(bins=50)
        plt.ylabel("Daily Return")
        plt.xlabel(None)
        plt.title(f"Daily return of {key}")
        i += 1

    plt.show()

########################################################################################################################
# PLOTLY
########################################################################################################################
def _plotly(title, name, colX, colY, data):
    layout = dict(title=title, xaxis=dict(title=colX), yaxis=dict(title=colY), plot_bgcolor="black")
    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")


def plotSomething_line(df, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", color=[COLOR_LINE],
                       loop: bool = False):
    x = df[colX].tolist()
    y = df[colY].tolist()

    data = go.Scatter(x=x, y=y, name=name, line=dict(color=color.pop(random.randint(0, len(color) - 1)), width=4))

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

    data = go.Scatter(x=x, y=y, name=colY, line=dict(color=color.pop(random.randint(0, len(color) - 1)),
                                                     width=2, dash="dash"))

    if not loop:
        # show
        title = "Plot {} on: {} - {}".format(name, colX, colY)
        _plotly(title, name, colX, colY, data)
    else:
        return data


def plotSomething_marker(df, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", color=COLOR_LINE,
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


def plotMultiple_line(datasets, colX: str = "Date", colY: str = "CloseUSD", name: str = "temp", color=COLORS):
    data = list()
    for key in datasets:
        df = datasets[key]
        data.append(plotSomething_line(df, colX, colY, key, color, True))

    # show
    title = "Plot {} on: {} - {}".format(name, colX, colY)
    _plotly(title, name, colX, colY, data)


def plotStocksTrend(datasets, colX: str = "Date", colY: str = "CloseUSD", name: str = "stocksTrend"):
    plotMultiple_line(datasets, colX, colY, name)


def plotStocksVolume(datasets, name: str = "temp"):
    plotMultiple_line(datasets, "Date", "Volume", name)


def _controlFeatures(df, cols):
    if cols is None:
        cols = STANDARD_FEATURE
        addFeatures(df, cols)
    return cols


def _setFeatures(df, col):
    if col.split("-")[0] == "BUY":
        return df[df[col] == 1], "green", "BuyPrice"
    elif col.split("-")[0] == "SELL":
        return df[df[col] == 1], "red", "SellPrice"


def plotStockFeatures(df, colX: str = "Date", colY: str = "CloseUSD", cols=None, name: str = "StockFeatureTemp",
                      dateStart=DATE_START, dateEnd=DATE_END):
    # subset con il limite delle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    cols = _controlFeatures(df, cols)

    # plot della stock
    title = "Plot {} on: {} - {}".format(name, colX, colY)
    color = COLORS

    data = list()
    data.append(plotSomething_line(df, colX, colY, name, [COLOR_LINE], True))

    # plot delle features
    for col in cols:
        if col.split("-")[0] == "BUY" or col.split("-")[0] == "SELL":
            dfPlot, colorPlot, colPlot = _setFeatures(df, col)
            data.append(plotSomething_marker(dfPlot, colX, colPlot, name, colorPlot, True))
        else:
            data.append(plotSomething_dash(df, colX, col, name, color, True))

        title += " - {}".format(col)

    # show
    _plotly(title, name, colX, colY, data)


def plotOhlc(df, name: str = "OhlcTemp", dateStart=DATE_START, dateEnd=DATE_END):
    # subset con il limite delle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    title = "Plot of {}".format(name)

    data = go.Ohlc(x=df.Date, open=df.Open, high=df.High, low=df.Low, close=df.CloseUSD)

    _plotly(title, name, "Date", "CloseUSD", data)


def plotCandlestick(df, name: str = "CandlestickTemp", dateStart=DATE_START, dateEnd=DATE_END):
    # subset con il limite delle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    title = "Plot of {}".format(name)

    data = go.Candlestick(x=df.Date, open=df.Open, high=df.High, low=df.Low, close=df.CloseUSD)

    _plotly(title, name, "Date", "CloseUSD", data)