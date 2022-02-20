import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "png"


def plotPie(*datasets, col: str = "Index"):
    for df in datasets:
        plt.figure(figsize=(10, 10))
        df[col].value_counts().plot.pie(autopct="%1.1f%%")
        plt.show()


def plotSomething(df, colX: str = "Date", colY: str = "Close"):
    fig = px.line(df, x=colX, y=colY)
    # # Vari setting per tittolo e colori extra
    # fig.update_layout(plot_bgcolor="black", title_text="CloseUSD for " + df["Index"].head())
    # fig.update_yaxes(showticklabels=True, showline=True, linewidth=2, linecolor="black")
    # fig.update_xaxes(showticklabels=True, showline=True, linewidth=2, linecolor="black")

    fig.show()


def plotStockTrend(df, colX: str = "Date", colY: str = "Close"):
    pass