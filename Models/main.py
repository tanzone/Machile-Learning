import pandas as pd

from DataAnalysis.basic import *
from DataAnalysis.utility import *
from DataAnalysis.plot import *


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
    # print(dfIndex)

    # # Utilizzo i dataset splittati per creare file csv ognuno per ogni stock
    # writeCsv_Index(groupByIndex(dfProc, takeIndex(dfProc)))

    # # Plot a torta della percentuale di indici che compaiono nel dataset
    # plotPie(dfProc)

    # # Plot grafico x y con Data e closeUSD di una stock
    # plotSomething(groupByIndex(dfProc, takeIndex(dfProc))["HSI"])

    # # prendo i vari indici e leggo il file correlato poi stampo il plot degli stocks
    # dfReaded= dict()
    # for i in takeIndex(dfProc):
    #    dfReaded[i] = pd.read_csv("../Dataset/"+i+".csv")

    # plotStocksTrend(dfReaded)

    # # Calcolo delle features aggiuntive e poi le plotto assieme al trend delle stock
    df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))["NYA"]

    # plotStocksTrend(df_TempPlot, "Date", "CloseUSD")

    features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "SMA-200", "SMA-500", "BUY-200-10", "SELL-5"]
    multipleFeature(df_TempPlot, features)
    plotStockFeatures(df_TempPlot, "Date", "CloseUSD", features, "NYA", "2020-01-01", "2021-01-01")






main()
