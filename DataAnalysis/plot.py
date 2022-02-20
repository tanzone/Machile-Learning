import random

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as ply

# Costante di colori per i plot
COLORS = ["red", "brown", "orange", "lawngreen", "green", "yellow", "aqua", "blue", "pink", "violet", "purple"]


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

    layout = dict(title="Plot {} on: {} - {}".format(name, colX, colY), xaxis=dict(title=colX), yaxis=dict(title=colY))

    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")
    if not save:
        # DA FARE, cancellare il file temp
        pass


def plotStocksTrend(datasets, colX: str = "Date", colY: str = "CloseUSD", name: str = "stocksTrend"):
    data = list()
    for key in datasets:
        df = datasets[key]
        x = df[colX].tolist()
        y = df[colY].tolist()
        data.append(
            go.Scatter(x=x, y=y, name=key, line=dict(color=COLORS[random.randint(0, len(COLORS) - 1)], width=4)))

    layout = dict(title="Plot {} on: {} - {}".format(name, colX, colY), xaxis=dict(title=colX), yaxis=dict(title=colY))

    fig = dict(data=data, layout=layout)
    ply.plot(fig, filename="../Plots/" + name + ".html")
