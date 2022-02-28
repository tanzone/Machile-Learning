from math import sqrt

from DataAnalysis.basic import *
from DataAnalysis.plot import *

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

SPLIT_CASUAL = "casual"
SPLIT_ALL = "all"
SPLIT_FINAL = "final"
SPLIT_DAYS = "days"

PRETYPE_MINMAX = "MinMax"
PRETYPE_STD = "Std"

FUTURE_DAYS = 10


def _split(df, splitType: str = SPLIT_CASUAL, size: float = 0.20, futureDays: int = FUTURE_DAYS):
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

    if splitType == SPLIT_DAYS:
        X_train = df.iloc[0: len(df) - futureDays].iloc[:, df.columns != "Close"]
        X_test = df.iloc[-futureDays:].iloc[:, df.columns != "Close"]
        y_train = df.iloc[0: len(df) - futureDays].iloc[:, df.columns == "Close"]
        y_test = df.iloc[-futureDays:].iloc[:, df.columns == "Close"]
        return X_train, y_train, X_test, y_test


def _preProcessing(X_train, y_train, X_test, y_test, preType: str = PRETYPE_MINMAX):
    listScaler = dict()
    if preType == PRETYPE_MINMAX:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        listScaler["X_train"] = scaler
        X_test = scaler.fit_transform(X_test)
        listScaler["X_test"] = scaler
        y_train = scaler.fit_transform(y_train)
        listScaler["y_train"] = scaler
        y_test = scaler.fit_transform(y_test)
        listScaler["y_test"] = scaler

        for i in listScaler:
            print(listScaler[i].min_, listScaler[i].max_)

        return X_train, y_train, X_test, y_test, listScaler

    #if preType == PRETYPE_STD:
        #avg = np.mean(X_train, axis=0)
        #std = np.std(X_train, axis=0)
        #X_train = (X_train - avg) / std
        #X_test = (X_test - avg) / std

        #return X_train, y_train, X_test, y_test


def _postProcessing(X_train, y_train, X_test, y_test, y_pred, scaler, xCols, yCols, postType: str = PRETYPE_MINMAX):
    if postType == PRETYPE_MINMAX:
        X_train = pd.DataFrame((scaler.pop(0)).inverse_transform(X_train), columns=xCols)
        X_test = pd.DataFrame((scaler.pop(0)).inverse_transform(X_test), columns=xCols)
        scalerY = (scaler.pop(0))
        y_train = pd.DataFrame(scalerY.inverse_transform(y_train), columns=yCols)
        y_test = pd.DataFrame((scaler.pop(0)).inverse_transform(y_test), columns=yCols)
        y_pred = pd.DataFrame(scalerY.inverse_transform(y_pred.reshape(-1, 1)), columns=yCols)

    # if postType == PRETYPE_STD:
        #avg = np.mean(X_train, axis=0)
        #std = np.std(X_train, axis=0)
        #X_train = (X_train - avg) / std
        #X_test = (X_test - avg) / std

    return X_train, y_train, X_test, y_test, y_pred


def _crossValidation(model, X_train, y_train, y_test, y_pred, name):
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2Score = r2_score(y_test, y_pred)
        print("Type of Regression: ".ljust(25) + name)
        # print("Coefficients: ".ljust(25), model.coef_)
        print("MRSE: ".ljust(25) + str(rmse))
        print("R2 Score: ".ljust(25), r2Score)
        print("CV Mean: ".ljust(25), np.mean(scores))
        print("STD: ".ljust(25), np.std(scores))
        print("------------------------------------------")

        return rmse


# modello regressione lineare semplice
def _linearRegression(X_train, y_train, X_test, y_test, name):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = _crossValidation(model, X_train, y_train, y_test, y_pred, name)

    return (y_pred, rmse)


def _polyRegression(X_train, y_train, X_test, y_test, num, name):
    poly = PolynomialFeatures(degree=num)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)

    return _linearRegression(X_train, y_train, X_test, y_test, name)


def _logisticRegression(X_train, y_train, X_test, y_test, name):
    model = LogisticRegression()
    label = LabelEncoder()
    y_train = label.fit_transform(y_train)
    model.fit(X_train, y_train)

    y_pred = label.inverse_transform(model.predict(X_test))

    rmse = 0#_crossValidation(model, X_train, y_train, y_test, y_pred, name)

    return (y_pred, rmse)


def modelUser(df, colDrop, split: str = SPLIT_CASUAL, size: float = 0.20,
              dateStart=DATE_START, dateEnd=DATE_END, plotTrain: bool = True):
    # Tolgo colonne che non voglio vendano prese dal machine learning
    df = df.drop(columns=colDrop)
    # Limitazione sulle date
    df = df.loc[(df["Date"] > dateStart) & (df["Date"] <= dateEnd)]
    # Trasformo la data da stringa a datetime cosicchè possa essere elaborata
    dateToTimeStamp(df)
    df["Date"] = df["Date"].astype('int64')

    # # Linear Regression
    # name = "Linear Regression"
    # Split
    # X_train, y_train, X_test, y_test = _split(df[:], split, size)
    # Learning
    # y_pred, rmse = _linearRegression(X_train[:], y_train[:], X_test[:], y_test[:], name)
    # Plot
    # plotCaso(df, X_train, y_train, X_test, y_test, y_pred, name, plotTrain)



    # # Polinomial Regression
    # Split
    # X_train, y_train, X_test, y_test = _split(df[:], split, size)
    # for i in range(1, 10):
    #     name = "Poly" + str(i)
    #     # Learning
    #     y_pred, rmse = _polyRegression(X_train[:], y_train[:], X_test[:], y_test[:], i, name)
    #     # Plot
    #     plotCaso(df, X_train, y_train, X_test, y_test, y_pred, name, plotTrain)



    # # Logistic Regression
    name = "Logistic Regression"
    preType = PRETYPE_MINMAX

    # Split
    X_train, y_train, X_test, y_test = _split(df[:], split, size)
    xCols = X_train.columns
    yCols = y_train.columns

    print(X_test)

    # PreProcessing
    X_train, y_train, X_test, y_test, listScaler = _preProcessing(X_train[:], y_train[:], X_test[:], y_test[:], preType)
    print(X_test)

    # Learning
    y_pred, rmse = _logisticRegression(X_train[:], y_train[:], X_test[:], y_test[:], name)

    # PostProcessing
    X_train, y_train, X_test, y_test, y_pred = _postProcessing(X_train[:], y_train[:], X_test[:], y_test[:], y_pred[:], listScaler, xCols, yCols, preType)
    print(X_test)

    # Plot
    plotCaso(df, X_train, y_train, X_test, y_test, y_pred, name, plotTrain)


def plotCaso(df, X_train, y_train, X_test, y_test, y_pred, name, plotTrain: bool = True):
    from matplotlib import pyplot as plt

    if plotTrain:
        plt.plot(df.Date, df.Close, label="Real")

        plotX, plotY = zip(*sorted(zip(X_train.Date, y_train.Close)))
        plt.scatter(plotX, plotY, label="Real-Train")

        plotX, plotY = zip(*sorted(zip(X_test.Date, y_test.Close)))
        plt.scatter(plotX, plotY, label="Real-Test")

    if not plotTrain:
        plotX, plotY = zip(*sorted(zip(X_test.Date, y_test.Close)))
        plt.plot(plotX, plotY, label="Real-Test")

    plotX, plotY = zip(*sorted(zip(X_test.Date, y_pred)))
    plt.plot(plotX, plotY, label=name)
    plt.legend()

    plt.show()


def main():
    stock = "HSI"
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")
    df = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # transformazione delle colonne in euro
    changeValue(df, stock, "GDAXI")

    # Primo Linear Regression su tutto il dataset con data e chiusura, predico punti casuali nel grafico
    # modelUser(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD", "Volume"], SPLIT_ALL)
    # modelUser(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD", "Volume"], SPLIT_ALL, 0.20, "2020-01-01", "2021-01-01")
    # # Primo Linear Regression su tutto il dataset con le features base, predico punti casuali nel grafico
    # modelUser(df[:], ["Adj Close", "CloseUSD"], SPLIT_CASUAL)
    # modelUser(df[:], ["Adj Close", "CloseUSD"], SPLIT_CASUAL, 0.20, "2020-01-01", "2021-01-01")
    # # Primo Linear Regression su tutto il dataset con le features avanzate, predico punti casuali nel grafico
    # Da fare--------------------


    # # Primo Linear Regression sull parte finale del dataset con data e chiusura, predico tot size della stock
    # modelUser(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD", "Volume"], SPLIT_FINAL)
    # modelUser(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD", "Volume"], SPLIT_FINAL, 0.20, "2020-01-01", "2021-01-01")
    # # Primo Linear Regression sull parte finale del dataset con le features base, predico tot size della stock
    # modelUser(df[:], ["Adj Close", "CloseUSD"], SPLIT_FINAL)
    # modelUser(df[:], ["Adj Close", "CloseUSD"], SPLIT_FINAL, 0.20, "2020-01-01", "2021-01-01")
    # # Primo Linear Regression sull parte finale del dataset le features avanzate, predico tot size della stock
    # Da fare-------------------


    # Primo Linear Regression su tutto il dataset con data e chiusura, predico tot giorni futuri
    # modelUser(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD", "Volume"], SPLIT_DAYS)
    modelUser(df[:], ["Open", "High", "Low", "Adj Close", "CloseUSD", "Volume"], SPLIT_DAYS, 0, "2020-01-01", "2021-01-01")
    # # Primo Linear Regression su tutto il dataset con le features base, predico tot giorni futuri
    # modelUser(df[:], ["Adj Close", "CloseUSD"], SPLIT_DAYS)
    # modelUser(df[:], ["Adj Close", "CloseUSD"], SPLIT_DAYS, 0, "2020-01-01", "2021-01-01")
    # Primo Linear Regression su tutto il dataset con le features avanzate, predico tot giorni futuri
    # Da fare--------------------





main()
