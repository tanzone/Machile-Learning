import pandas as pd

from DataAnalysis.utility import *


def info(*datasets):
    for df in datasets:
        df.info()
        print("------------------------------------------")


def countValue(*datasets, col: str = "Index"):
    for df in datasets:
        print(df[col].value_counts())
        print("------------------------------------------")

def main():
    dfData = pd.read_csv("../Dataset/indexData.csv")
    dfInfo = pd.read_csv("../Dataset/indexInfo.csv")
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")

    # # info sui dataset
    # info(dfData, dfInfo, dfProc)
    # # numero valori per indice
    # countValue(dfData, dfProc)

    # Mixo data e info
    dfTot = dfData.merge(dfInfo, on="Index", how="inner")

    listIndex = list()
    listIndex = dfData["Index"].unique()
    print(listIndex)


    #print(dfTot)



main()
