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
    dfIndex = groupByIndex(dfProc, takeIndex(dfProc))

    # # Utilizzo i dataset splittati per creare file csv ognuno per ogni stock
    # writeCsv_Index(groupByIndex(dfProc, takeIndex(dfProc)))

    # # Plot a torta della percentuale di indici che compaiono nel dataset
    # plotPie(dfProc)

    # # Plot grafico x y con Data e closeUSD di una stock
    plotSomething(dfIndex["HSI"])






main()
