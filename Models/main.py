import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px

from DataAnalysis.utility import *


def info(*datasets):
    for df in datasets:
        df.info()
        print("------------------------------------------\n")


def controlValues(*datasets):
    for df in datasets:
        print(df.isnull().any(), end="\n------------------\n")
        print(df.dtypes, end="\n------------------\n")


def countValue(*datasets, col: str = "Index"):
    for df in datasets:
        print(df[col].value_counts())
        print("------------------------------------------")


def takeIndex(df):
    return df["Index"].unique()


def groupByIndex(df, index):
    dfIndex = dict()
    for i in index:
        dfIndex[i] = df.groupby(df.Index).get_group(i).reset_index().drop(["Index", "index"], axis=1)

    return dfIndex


def writeCsv_Index(df):
    for key in df:
        df[key].to_csv("../Dataset/" + key + ".csv", index=False)


def plotPie(*datasets, col: str = "Index"):
    for df in datasets:
        plt.figure(figsize=(10, 10))
        df[col].value_counts().plot.pie(autopct="%1.1f%%")
        plt.show()


def plotSomething(df, colX: str = "Date", colY: str = "Close"):
    fig = px.line(df[["Date", "Close"]], x=colX, y=colY)
    # # Vari setting per tittolo e colori extra
    # fig.update_layout(plot_bgcolor="black", title_text="CloseUSD for " + df["Index"].head())
    # fig.update_yaxes(showticklabels=True, showline=True, linewidth=2, linecolor="black")
    # fig.update_xaxes(showticklabels=True, showline=True, linewidth=2, linecolor="black")

    fig.show()


def main():
    dfData = pd.read_csv("../Dataset/indexData.csv")
    dfInfo = pd.read_csv("../Dataset/indexInfo.csv")
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")

    # # info sui dataset
    # info(dfData, dfInfo, dfProc)
    # controlValues(dfData, dfInfo, dfProc)
    # # numero valori per indice
    # countValue(dfData, dfProc)

    # # Mixo data e info
    # dfTot = dfData.merge(dfInfo, on="Index", how="inner")
    # print(dfTot)

    # # Prendo tutti gli indice degli stocks nel dataset processed perchè
    # # possiede la colonna CloseUSD quindi unificata per tutti gli indici
    # # e divido il dataset nei vari subset in base all'indice della stock
    # dfIndex = groupByIndex(dfProc, takeIndex(dfProc))

    # # Utilizzo i dataset splittati per creare file csv ognuno per ogni stock
    # writeCsv_Index(groupByIndex(dfProc, takeIndex(dfProc)))

    # # Plot a torta della percentuale di indici che compaiono nel dataset
    # plotPie(dfProc)

    # # Plot grafico x y con Data e closeUSD di una stock
    # plotSomething(dfIndex[listIndex[0]])






main()
