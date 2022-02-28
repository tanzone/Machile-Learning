from DataAnalysis.basic import *
from DataAnalysis.plot import *

from DataAnalysis.utility import *


def main():
    dfData = pd.read_csv("../Dataset/indexData.csv")
    dfInfo = pd.read_csv("../Dataset/indexInfo.csv")
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")


    # # info sui dataset
    # info(dfData, dfInfo, dfProc)
    # controlValues(dfData, dfInfo, dfProc)
    #
    # # numero valori per indice
    # countValue(dfData, dfProc)
    #
    #
    # # Mixo data e info
    # dfTot = dfData.merge(dfInfo, on="Index", how="inner")
    # print(dfTot)
    #
    #
    # # Prendo tutti gli indice degli stocks nel dataset processed perchè
    # # possiede la colonna CloseUSD quindi unificata per tutti gli indici
    # # e divido il dataset nei vari subset in base all'indice della stock
    # dfIndex = groupByIndex(dfProc, takeIndex(dfProc))
    # print(dfIndex)
    #
    #
    # # Utilizzo i dataset splittati per creare file csv ognuno per ogni stock
    # writeCsv_Index(groupByIndex(dfProc, takeIndex(dfProc)))
    #
    #
    # # Plot a torta della percentuale di indici che compaiono nel dataset
    # plotPie(dfProc)
    #
    #
    # # Plot a barre della quantità di valori presenti per giorni della settimana
    # plotBar(dfProc)
    # stock = "HSI"
    # plotBar(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    #
    #
    # # Plot grafico x y con Data e closeUSD di una stock
    # stock = "HSI"
    # plotSomething_line(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    #
    #
    # # Plot grafico x y con data e volume
    # stock = "NYA"
    # plotSomething_line(groupByIndex(dfProc, takeIndex(dfProc))[stock], "Date", "Volume", "Volume of " + stock)
    #
    #
    # # prendo i vari indici e leggo il file correlato poi stampo il plot degli stocks
    # dfReaded= dict()
    # for i in takeIndex(dfProc):
    #     dfReaded[i] = pd.read_csv("../Dataset/"+i+".csv")
    # plotStocksTrend(dfReaded)
    #
    #
    # # Multi plot dei vari trend degli stock con grafico dedicato
    # plotStocksTrend_matlib(groupByIndex(dfProc, takeIndex(dfProc)))
    # # Multi plot dei vari trend degli stock con varie features
    # plotStocksTrend_matlib(groupByIndex(dfProc, takeIndex(dfProc)), STANDARD_FEATURE_MATLIB)
    #
    #
    # # Multi plot del ritorno giornaliero delle stock
    # plotStocksReturn_matlib(groupByIndex(dfProc, takeIndex(dfProc)), 1)
    #
    # # Multi plot del ritorno giornaliero delle stock come media a grafico a barre
    # plotStocksReturn_matlib_bar(groupByIndex(dfProc, takeIndex(dfProc)), 1)
    #
    #
    # # Plot volume delle stocks
    # dfReaded = dict()
    # for i in takeIndex(dfProc):
    #    dfReaded[i] = pd.read_csv("../Dataset/"+i+".csv")
    # plotStocksVolume(dfReaded)
    #
    #
    # # plotto le feature assieme al trend della stock
    # stock = "NYA"
    # plotStockFeatures(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    #
    #
    # # Calcolo delle features aggiuntive e poi le plotto assieme al trend della stock
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "SMA-200", "SMA-500", "EMA-5", "EMA-50", "LOW-10", "HIGH-10",
    #             "CUMCHANGE-1", "EXPANDING-MEAN", "EXPANDING-STD", "BUY-200-10", "SELL-5"]
    # addFeatures(df_TempPlot, features)
    # plotStockFeatures(df_TempPlot, "Date", "CloseUSD", features, "NYA")
    # plotStockFeatures(df_TempPlot, "Date", "CloseUSD", features, "NYA", "2020-01-01", "2021-01-01")
    #
    #
    # # Plot (matplotlib) tutte le info su una stock
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # plotInfoStock(df_TempPlot)
    #
    #
    # # Plot (matplotlib) tutte le info su una stock con l'aggiunta di altre features
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "SMA-200", "SMA-500", "LOW-10", "HIGH-10",
    #             "EXPANDING-MEAN", "EXPANDING-STD", "BUY-200-10", "SELL-5"]
    # addFeatures(df_TempPlot, features)
    # plotInfoStock(df_TempPlot)
    #
    #
    # # Plot Outliers
    # plotOutliers(groupByIndex(dfProc, takeIndex(dfProc)), "CloseUSD")
    #
    #
    # # Plot dell'Ohlc di una stock
    # stock = "NYA"
    # plotOhlc(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    # plotOhlc(groupByIndex(dfProc, takeIndex(dfProc))[stock], "Ohlc-" + stock, "2020-01-01", "2021-01-01")
    #
    #
    # # Plot di Candlestick di una stock
    # stock = "NYA"
    # plotCandlestick(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    # plotCandlestick(groupByIndex(dfProc, takeIndex(dfProc))[stock], "Candlestick-" + stock, "2020-01-01", "2021-01-01")
    #
    #
    # # ADF test per la stazionarietà
    # toTest = dfProc[["Date", "CloseUSD"]].copy()
    # toTest.set_index("Date", inplace=True)
    # toTest.index = pd.to_datetime(toTest.index)
    # toTest = toTest.resample('1M').mean()
    # adfTest(toTest["CloseUSD"], title='')
    #
    #
    # # Plot volatilità delle stock
    # plotVolatility(groupByIndex(dfProc, takeIndex(dfProc)))
    #
    #
    # # Plot heatMap di una stock con le features base
    # stock = "NYA"
    # plotHeatMap_features(groupByIndex(dfProc, takeIndex(dfProc))[stock])
    #
    #
    # # Plot heatMap di una stock con features aggiuntive
    # stock = "NYA"
    # df_TempPlot = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "SMA-200", "SMA-500", "LOW-10", "HIGH-10",
    #             "EXPANDING-MEAN", "EXPANDING-STD", "BUY-200-10", "SELL-5"]
    # addFeatures(df_TempPlot, features)
    # plotHeatMap_features(df_TempPlot)
    #
    #
    # # Plot heatMap delle stock tra loro
    # plotHeatMap_stock(groupByIndex(dfProc, takeIndex(dfProc)), "CloseUSD", "2020-01-01", "2021-01-01")
    #
    #
    # # Plot joint delle stock a due a due
    # plotJoint(groupByIndex(dfProc, takeIndex(dfProc)), "NYA", "NYA", "CloseUSD", "2020-01-01", "2021-01-01")
    # plotJoint(groupByIndex(dfProc, takeIndex(dfProc)), "NYA", "HSI", "CloseUSD", "2020-01-01", "2021-01-01")
    # plotJoint(groupByIndex(dfProc, takeIndex(dfProc)), "000001.SS", "399001.SZ", "CloseUSD", "2020-01-01", "2021-01-01")
    # plotJoint(groupByIndex(dfProc, takeIndex(dfProc)), "HSI", "GDAXI", "CloseUSD", "2020-01-01", "2021-01-01")
    #
    #
    # # Plot pair tra tutte le stocks
    # plotPair(groupByIndex(dfProc, takeIndex(dfProc)), "CloseUSD", "2020-01-01", "2021-01-01")
    #
    #
    # # Plot dettagli
    # # Da errori perchè c'è 0 varianza tra alcune combinazioni di stocks ma non è un problema
    # plotDetails(groupByIndex(dfProc, takeIndex(dfProc)), "CloseUSD", "2020-01-01", "2021-01-01")
    #
    #
    # # Plot Risk per stocks
    # plotRisk(groupByIndex(dfProc, takeIndex(dfProc)), "CloseUSD", "2020-01-01", "2021-01-01")
    #
    #
    # # Calcolo covarianza delle stock fra loro
    # covMatrix(groupByIndex(dfProc, takeIndex(dfProc)), "CHANGE-1")
    #
    #
    # # Calcolo correlazione delle stock fra loro
    # corrMatrix(groupByIndex(dfProc, takeIndex(dfProc)), "CHANGE-1")
    #
    #
    # # Conversione del valore della moneta in ogni combinazione
    # print(getAllCurrency(groupByIndex(dfProc, takeIndex(dfProc))))
    #
    # # Convesione di tutte le colonne di un df nella moneta che desidero
    # stock = "000001.SS"
    # changeValue(groupByIndex(dfProc, takeIndex(dfProc))[stock], stock, "NYA")


if __name__ == "__main__":
    main()
