from datetime import datetime

from DataAnalysis.utility import *


def info(*datasets):
    for df in datasets:
        df.info()
        print("------------------------------------------\n")
        print(df.describe())
        print("------------------------------------------\n")
        print(df.groupby('Index').describe())
        print("------------------------------------------\n")
        print(df.isnull().sum())
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


def getAllCurrency(datasets):
    allRates = {}
    rates = {}
    for key in datasets:
        df = datasets[key]
        rates[key + "_USD"] = df.Close[0] / df.CloseUSD[0]

    for key in datasets:
        df = datasets[key]
        for k in rates:
            name = k.split("_")[0]
            df["Close" + name] = df.CloseUSD * rates[k]

    for k in rates:
        r = 1 / rates[k]
        for j in rates:
            start = k.split("_")[0]
            end = j.split("_")[0]
            allRates[start + "_" + end] = float(r) * float(rates[j])

    return allRates


def changeValue(df, moneyCurr, moneyTran):
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")
    rates = getAllCurrency(groupByIndex(dfProc, takeIndex(dfProc)))
    change = rates[moneyCurr + "_" + moneyTran]

    df["Open"] = df.Open * change
    df["High"] = df.High * change
    df["Low"] = df.Low * change
    df["Close"] = df.Close * change
    df["Adj Close"] = df["Adj Close"] * change


def dateToTimeStamp(df):
    dateList = list()
    for date in df["Date"]:
        dateList.append(datetime.strptime(date, "%Y-%m-%d").timestamp())

    df["Date"] = pd.Series(dateList).values


def timeStampToDate(df):
    dateList = list()
    for date in df["Date"]:
        dateList.append(datetime.fromtimestamp(date).date().strftime("%Y-%m-%d"))

    df["Date"] = pd.Series(dateList).values


def timeStampToDate_list(array):
    dateList = list()
    for date in array:
        dateList.append(datetime.fromtimestamp(date).date().strftime("%Y-%m-%d"))

    return dateList

