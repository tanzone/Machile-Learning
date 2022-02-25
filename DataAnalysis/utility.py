import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# change percentuale calcolato su un periodo
def rollingChange(df, col: str = "High", num: int = 30):
    df["CHANGE"] = df["CloseUSD"].pct_change(periods = num)


def rollingReturn(df, col: str = "CloseUSD", num: int = 30):
    df["RETURN"] = (df[col] / df[col].shift(1)) - 1


def stockCumReturn(df):
    rollingReturn(df, "CloseUSD")
    df["CUMRETURN"] = (1 + df["RETURN"]).cumprod()


def rollingSma(df, col: str = "CloseUSD", num: int = 5):
    df["SMA-" + str(num)] = df[col].rolling(num).mean()


#Da provare
def rollingEma(df, col: str = "CloseUSD", num: int = 5):
    df["EMA-" + str(num)] = df[col].ewm(span=num, adjust=False).mean()


#da finire
def rollingRoi():
    pass


#da finire
def rollingRoe():
    pass


#da finire
def rollingEps():
    pass


#da finire
def rollingCagr():
    pass



def rollingLow(df, col: str = "CloseUSD", num: int = 10):
    df["LOW-" + str(num)] = df[col].rolling(num).min()


def rollingHigh(df, col: str = "CloseUSD", num: int = 10):
    df["HIGH-" + str(num)] = df[col].rolling(num).max()


def expandingMean(df, col: str = "High"):
    df["EXPANDING-MEAN"] = df[col].expanding().mean()


def expandingStd(df, col: str = "High"):
    df["EXPANDING-STD"] = df[col].expanding().std()


#da sistemare
def buyTime(df, col1: str = "CloseUSD", sma1: str = "SMA-200", col2: str = "LOW-10", rate: float = 0.98):
    # creo lo sma-? necessario per il calcolo finale
    rollingSma(df, col1, int(sma1.split("-")[1]))
    # creo il low-? necessario per il calcolo finale
    rollingLow(df, col1, int(col2.split("-")[1]))

    df["BUY-" + sma1.split("-")[1] + "-" + col2.split("-")[1]] = np.where((df[col1] > df[sma1]) &
                                                                          (df[col2].diff() < 0) &
                                                                          (df[col1] * rate >= df["Low"].shift(-1)), 1,
                                                                          0)

    df["BuyPrice"] = rate * df[col1]


#da sistemare
def sellTime(df, col1: str = "CloseUSD", sma1: str = "SMA-5", col2: str = "CloseUSD"):
    # creo lo sma-? necessario per il calcolo finale
    rollingSma(df, col1, int(sma1.split("-")[1]))
    df["SELL-" + sma1.split("-")[1]] = np.where((df[col1] > df[sma1]), 1, 0)
    df["SellPrice"] = df[col2].shift(-1)  # DA FARE, capire se ci vuole o meno lo shift


#da rivedere
def profits(df):
    dfProfits = df.copy()

    rollingSma(dfProfits, "CloseUSD", 5)
    rollingSma(dfProfits, "CloseUSD", 200)
    rollingLow(dfProfits, "CloseUSD", 10)
    dfProfits.dropna(inplace=True)

    buyTime(dfProfits, "CloseUSD", "SMA-200", "LOW-10", 0.98)
    sellTime(dfProfits, "SMA-5")

    x = df[(dfProfits.BUY == 1) or (dfProfits.SELL == 1)]
    y = x[(x.BUY.diff() == 1) or (x.SELL.diff() == 1)]

    return ((y.SellPrice.shift(-1) - y.BuyPrice) / y.BuyPrice)[::2]


def backTest(arr):
    winrate = len(arr[arr > 0]) / len(arr)
    max_dd = min(arr)
    mean = arr.mean()
    gain = (arr + 1).cumprod()[-1]

    return winrate, max_dd, mean, gain


def addFeatures(df, features):
    for feature in features:
        if feature.split("-")[0] == "SMA":
            rollingSma(df, "CloseUSD", int(feature.split("-")[1]))
        if feature.split("-")[0] == "LOW":
            rollingLow(df, "CloseUSD", int(feature.split("-")[1]))
        if feature.split("-")[0] == "HIGH":
            rollingHigh(df, "CloseUSD", int(feature.split("-")[1]))
        if feature.split("-")[0] == "BUY":
            buyTime(df, "CloseUSD", "SMA-" + feature.split("-")[1], "LOW-" + feature.split("-")[2])
        if feature.split("-")[0] == "SELL":
            sellTime(df, "CloseUSD", "SMA-" + feature.split("-")[1], "Open")
        if feature.split("-")[0] == "CHANGE":
            rollingChange(df, "High", int(feature.split("-")[1]))
        if feature.split("-")[0] == "EXPANDING":
            if feature.split("-")[1] == "MEAN":
                expandingMean(df, "High")
            elif feature.split("-")[1] == "STD":
                expandingStd(df, "High")


def adfTest(series, title: str = ""):
    print("Augmented Dickey-Fuller Test: {}".format(title))
    result = adfuller(series.dropna(), autolag="AIC")

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out["critical value ({})".format(key)] = val

    print(out.to_string())

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
