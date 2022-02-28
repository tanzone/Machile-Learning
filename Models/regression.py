from math import sqrt

from DataAnalysis.basic import *
from DataAnalysis.plot import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

SPLIT_CASUAL = "casual"
SPLIT_ALL = "all"
SPLIT_FINAL = "final"

FUTURE_DAYS = 30


def _split(df, splitType: str = SPLIT_CASUAL, size: float = 0.20):
    if splitType == SPLIT_CASUAL:
        X = df.iloc[:, df.columns != "Close"]
        y = df.iloc[:, df.columns == "Close"]
        return train_test_split(X, y, test_size=size)

    if splitType == SPLIT_ALL:
        X = df.iloc[:, df.columns != "Close"]
        y = df.iloc[:, df.columns == "Close"]
        return train_test_split(X, y, test_size=size, random_state=27)

    if splitType == SPLIT_FINAL:
        X_train = df.iloc[:int(len(df) * (1-size))].iloc[:, df.columns != "Close"]
        X_test  = df.iloc[int(len(df) * (1-size)):].iloc[:, df.columns != "Close"]
        y_train = df.iloc[:int(len(df) * (1-size))].iloc[:, df.columns == "Close"]
        y_test  = df.iloc[int(len(df) * (1-size)):].iloc[:, df.columns == "Close"]
        return X_train, X_test, y_train, y_test


# poco sensato fatto cosi
def primoLinearRegression(df, colDrop, split: str = SPLIT_CASUAL, size: float = 0.20,
                          dateStart=DATE_START, dateEnd=DATE_END):
    # Tolgo colonne che non voglio vendano prese dal machine learning
    df = df.drop(columns=colDrop)
    # Limitazione sulle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    # Trasformo la data da stringa a datetime cosicchè possa essere elaborata
    dateToTimeStamp(df)

    # Split
    X_train, X_test, y_train, y_test = _split(df, split, size)

    # modello regressione lineare
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(rmse)

    plotCaso(df, X_train, X_test, y_train, y_test, y_pred)


def plotCaso(df, X_train, X_test, y_train, y_test, y_pred):
    from matplotlib import pyplot as plt
    plt.plot(df.Date, df.Close, label="Real")

    plotX, plotY = zip(*sorted(zip(X_train.Date, y_train.Close)))
    plt.scatter(plotX, plotY, label="Real-Train")

    plotX, plotY = zip(*sorted(zip(X_test.Date, y_test.Close)))
    plt.scatter(plotX, plotY, label="Real-Test")

    plotX, plotY = zip(*sorted(zip(X_test.Date, y_pred)))
    plt.plot(plotX, plotY, label="Linear Regression")
    plt.legend()

    plt.show()


def main():
    stock = "HSI"
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")
    df = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # transformazione delle colonne in euro
    changeValue(df, stock, "GDAXI")

    # Primo Linear Regression su tutto il dataset con data e chiusura
    # primoLinearRegression(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD"], SPLIT_ALL)
    # primoLinearRegression(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD"], SPLIT_ALL, 0.20, "2020-01-01", "2021-01-01")
    # # Primo Linear Regression su tutto il dataset con le features base
    # primoLinearRegression(df[:], ["Adj Close", "CloseUSD"], SPLIT_CASUAL)
    # primoLinearRegression(df[:], ["Adj Close", "CloseUSD"], SPLIT_CASUAL, 0.20, "2020-01-01", "2021-01-01")
    # Primo Linear Regression su tutto il dataset con le features avanzate
    # Da fare--------------------

    # Primo Linear Regression sull parte finale del dataset con data e chiusura (Realistico)
    # primoLinearRegression(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD"], SPLIT_FINAL)
    primoLinearRegression(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD"], SPLIT_FINAL, 0.20, "2020-01-01", "2021-01-01")
    # Primo Linear Regression sull parte finale del dataset con le features base
    # primoLinearRegression(df[:], ["Adj Close", "CloseUSD"], SPLIT_FINAL)
    # primoLinearRegression(df[:], ["Adj Close", "CloseUSD"], SPLIT_FINAL, 0.20, "2020-01-01", "2021-01-01")
    # Primo Linear Regression sull parte finale del dataset le features avanzate
    # Da fare-------------------





main()
