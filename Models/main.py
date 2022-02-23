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
    # stock = "HSI"
    # plotSomething(groupByIndex(dfProc, takeIndex(dfProc))[stock])


    # # Plot grafico x y con data e volume
    # stock = "NYA"
    # plotSomething_line(groupByIndex(dfProc, takeIndex(dfProc))[stock], "Date", "Volume", "Volume of " + stock)


    # # prendo i vari indici e leggo il file correlato poi stampo il plot degli stocks
    # dfReaded= dict()
    # for i in takeIndex(dfProc):
    #    dfReaded[i] = pd.read_csv("../Dataset/"+i+".csv")
    # plotStocksTrend(dfReaded)


    # # Multi plot dei vari trend degli stock con grafico dedicato
    # plotStocksTrend_matlib(groupByIndex(dfProc, takeIndex(dfProc)))
    # # Multi plot dei vari trend degli stock con varie features
    # plotStocksTrend_matlib(groupByIndex(dfProc, takeIndex(dfProc)), STANDARD_FEATURE_MATLIB)

    # # Multi plot del ritorno giornaliero delle stock
    # plotStocksReturn_matlib(groupByIndex(dfProc, takeIndex(dfProc)))

    # # Multi plot del ritorno giornaliero delle stock come media a grafico a barre
    plotStocksReturn_matlib_bar(groupByIndex(dfProc, takeIndex(dfProc)))

    # # Plot volume delle stocks
    # dfReaded = dict()
    # for i in takeIndex(dfProc):
    #    dfReaded[i] = pd.read_csv("../Dataset/"+i+".csv")
    # plotStocksVolume(dfReaded)


    # # plotto le feature assieme al trend della stock
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # plotStockFeatures(df_TempPlot)


    # # Calcolo delle features aggiuntive e poi le plotto assieme al trend della stock
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "SMA-200", "SMA-500", "LOW-10", "HIGH-10",
    #             "EXPANDING-MEAN", "EXPANDING-STD", "BUY-200-10", "SELL-5"]
    # addFeatures(df_TempPlot, features)
    # plotStockFeatures(df_TempPlot, "Date", "CloseUSD", features, "NYA")
    # plotStockFeatures(df_TempPlot, "Date", "CloseUSD", features, "NYA", "2020-01-01", "2021-01-01")


    # # Plot (matplotlib) tutte le info su una stock
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # plotInfoStock(df_TempPlot)


    # # Plot dell'Ohlc di una stock
    # stock = "NYA"
    # plotOhlc(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    # plotOhlc(groupByIndex(dfProc, takeIndex(dfProc))[stock], "Ohlc-" + stock, "2020-01-01", "2021-01-01")


    # # Plot di Candlestick di una stock
    # stock = "NYA"
    # plotCandlestick(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    # plotCandlestick(groupByIndex(dfProc, takeIndex(dfProc))[stock], "Candlestick-" + stock, "2020-01-01", "2021-01-01")







if __name__ == "__main__":
    main()
