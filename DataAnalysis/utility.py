import numpy as np


def rollingMean(df, col: str = "Close", num: int = 5):
    df["SMA-" + str(num)] = df[col].rolling(num).mean()


def rollingLow(df, col: str = "Close", num: int = 10):
    df["LOW-" + str(num)] = df[col].rolling(num).min()


def rollingHigh(df, col: str = "Close", num: int = 10):
    df["HIGH-" + str(num)] = df[col].rolling(num).max()


def buyTime(df, col1: str = "Close", sma1: str = "SMA-200", col2: str = "LOW-10", rate: float = 0.98):
    # creo lo sma-? necessario per il calcolo finale
    rollingMean(df, col1, int(sma1.split("-")[1]))
    # creo il low-? necessario per il calcolo finale
    rollingLow(df, col1, int(col2.split("-")[1]))

    df["BUY"] = np.where((df[col1] > df[sma1]) &
                         (df[col2].diff() < 0) &
                         (df[col1] * rate >= df["Low"].shift(-1)), 1, 0)

    df["BuyPrice"] = rate * df[col1]


def sellTime(df, col1: str = "Close", sma1: str = "SMA-5", col2: str = "Open"):
    # creo lo sma-? necessario per il calcolo finale
    rollingMean(df, col1, int(sma1.split("-")[1]))
    df["SELL"] = np.where((df[col1] > df[sma1]), 1, 0)
    df["SellPrice"] = df[col2].shift(-1)


def profits(df):
    dfProfits = df.copy()

    rollingMean(dfProfits, "Close", 5)
    rollingMean(dfProfits, "Close", 200)
    rollingLow(dfProfits, "Close", 10)
    dfProfits.dropna(inplace=True)

    buyTime(dfProfits, "Close", "SMA-200", "LOW-10", 0.98)
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






##
# rollingMean(df, "Close", 5)
# rollingMean(df, "Close", 30)
# rollingMean(df, "Close", 100)
# rollingMean(df, "Close", 200)
#
# rollingLow(df, "Close", 10)
# rollingHigh(df, "Close", 10)
# rollingLow(df, "Close", 100)
# rollingHigh(df, "Close", 100)
#
# df.dropna(inplace=True)
#
# buyTime(df, "Close", "SMA-200", "LOW-10", 0.98)
# sellTime(df, "SMA-5")
#
# # backtest di Rayner Teo
# backTest(profits(df))
##

