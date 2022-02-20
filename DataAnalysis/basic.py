import numpy as np
import pandas as pd


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