import numpy as np
import pandas as pd

from DataAnalysis.plot import DATE_START, DATE_END

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

from math import sqrt


DROP_ALL = ["Adj Close", "CloseUSD"]
DROP_WASTE = ["Adj Close", "CloseUSD", "Open", "High", "Low", "Volume"]

MODIFY_ALL_ALL = DROP_ALL, DATE_START, DATE_END
MODIFY_ALL_YEAR = DROP_ALL, "2020-01-01", "2021-01-01"
MODIFY_WASTE_ALL = DROP_WASTE, DATE_START, DATE_END
MODIFY_WASTE_YEAR = DROP_WASTE, "2020-01-01", "2021-01-01"

SPLIT_ALL = "all"
SPLIT_ALL_CASUAL = "casual"
SPLIT_FINAL_SIZE = "final"
SPLIT_FINAL_DAYS = "days"
FUTURE_DAYS = 50

MODEL_1 = "Linear Regression"
MODEL_2 = "PolynomialFeatures Regression"
MODEL_3 = "Logistic Regression"
MODEL_4 = ""
MODEL_5 = ""
MODEL_6 = ""

# TODO da aggiungerci che tipo di normalizzazione vuole il modello
MODELS = dict()
MODELS[MODEL_1] = True
MODELS[MODEL_2] = False
MODELS[MODEL_3] = True
MODELS[MODEL_4] = True
MODELS[MODEL_5] = True
MODELS[MODEL_6] = True


PRETYPE_FALSE = "False"
PRETYPE_MINMAX = "MinMax"
PRETYPE_STD = "Std"


# Vari tipi di split
def _split(df, splitType: str = SPLIT_ALL_CASUAL, size: float = 0.20, futureDays: int = FUTURE_DAYS):
    if splitType == SPLIT_ALL_CASUAL:
        X = df.iloc[:, df.columns != "Close"]
        y = df.iloc[:, df.columns == "Close"]
        return train_test_split(X, y, test_size=size)

    if splitType == SPLIT_ALL:
        X = df.iloc[:, df.columns != "Close"]
        y = df.iloc[:, df.columns == "Close"]
        return train_test_split(X, y, test_size=size, random_state=27)

    if splitType == SPLIT_FINAL_SIZE:
        X_train = df.iloc[:int(len(df) * (1 - size))].iloc[:, df.columns != "Close"]
        X_test = df.iloc[int(len(df) * (1 - size)):].iloc[:, df.columns != "Close"]
        y_train = df.iloc[:int(len(df) * (1 - size))].iloc[:, df.columns == "Close"]
        y_test = df.iloc[int(len(df) * (1 - size)):].iloc[:, df.columns == "Close"]
        return X_train, X_test, y_train, y_test

    if splitType == SPLIT_FINAL_DAYS:
        X_train = df.iloc[0: len(df) - futureDays].iloc[:, df.columns != "Close"]
        X_test = df.iloc[-futureDays:].iloc[:, df.columns != "Close"]
        y_train = df.iloc[0: len(df) - futureDays].iloc[:, df.columns == "Close"]
        y_test = df.iloc[-futureDays:].iloc[:, df.columns == "Close"]
        return X_train, y_train, X_test, y_test


# TODO altri tipi
# Prepocessing di normalizzazione e standizzazione
def _preProcessing(X_train, X_test, preType: str = PRETYPE_MINMAX):
    if preType == PRETYPE_FALSE:
        return X_train, X_test, None

    scaler = None

    if preType == PRETYPE_MINMAX:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if preType == PRETYPE_STD:
        avg = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - avg) / std
        X_test = (X_test - avg) / std

    return X_train, X_test, scaler


# TODO altri tipi in catena col preprocessing
# Postpocessing di normalizzazione e standizzazione
def _postProcessing(X_train, X_test, scaler, xCols, preType: str = PRETYPE_MINMAX):
    if preType == PRETYPE_FALSE:
        return X_train, X_test

    if preType == PRETYPE_MINMAX:
        X_train = pd.DataFrame(scaler.inverse_transform(X_train), columns=xCols)
        X_test = pd.DataFrame(scaler.inverse_transform(X_test), columns=xCols)

    # TODO da fare il contrario
    if preType == PRETYPE_STD:
        avg = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - avg) / std
        X_test = (X_test - avg) / std

    return X_train, X_test


# TODO verificare che ci siano tutti
# Vari indicatori di validazione per la regressione
def _crossValidation(model, X_train, y_train, y_test, y_pred, name, cv=5, scoring="r2"):
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
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


# TODO da metterlo a post e nella giusta cartella
# TODO plottare anche con l'altro sistema
def plotCaso(df, infoModel, name, plotTrain: bool = True):
    from matplotlib import pyplot as plt

    if plotTrain:
        plt.plot(df.Date, df.Close, label="Real")

        plotX, plotY = zip(*sorted(zip(infoModel[0].Date, infoModel[1].Close)))
        plt.scatter(plotX, plotY, label="Real-Train")

        plotX, plotY = zip(*sorted(zip(infoModel[2].Date, infoModel[3].Close)))
        plt.scatter(plotX, plotY, label="Real-Test")

    if not plotTrain:
        plotX, plotY = zip(*sorted(zip(infoModel[2].Date, infoModel[3].Close)))
        plt.plot(plotX, plotY, label="Real-Test")

    plotX, plotY = zip(*sorted(zip(infoModel[2].Date, infoModel[4])))
    plt.plot(plotX, plotY, label=name)
    plt.legend()

    plt.show()

