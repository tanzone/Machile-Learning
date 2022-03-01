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
FUTURE_DAYS = 5

MODEL_1 = "Linear Regression"
MODEL_2 = "PolynomialFeatures Regression"
MODEL_3 = "Logistic Regression"
MODEL_4 = "RandomForest Regression"
MODEL_5 = "ADABoosting Regression"
MODEL_6 = "GradientBoosting Regression"

PRETYPE_FALSE = "False"
PRETYPE_MINMAX = "MinMax"
PRETYPE_STD = "Std"

KFOLD_NUM = 10


MODELS = dict()
MODELS[MODEL_1] = {"Active": False,  "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": False, "crossType": False}

MODELS[MODEL_2] = {"Active": False,  "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": False, "crossType": False}

MODELS[MODEL_3] = {"Active": False,  "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False}

MODELS[MODEL_4] = {"Active": True,  "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True}

MODELS[MODEL_5] = {"Active": False,  "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True}

MODELS[MODEL_6] = {"Active": False,  "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True}




RANDOM_FOREST_REGRESSION_SPACE = {"n_estimators": [int(x) for x in np.linspace(start=1, stop=200, num=15)],
                                  "max_features": ["auto", "sqrt"], "bootstrap": [True, False],
                                  "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                                  "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}


ADABOOST_REGRESSION_SPACE = {"n_estimators": [int(x) for x in np.linspace(start=0, stop=201, num=50)],
                             "learning_rate": [float(x/100) for x in np.linspace(start=0, stop=20, num=3)],
                             "loss": ["linear", "square", "exponential"]}

GRADIENTBOOST_REGRESSION_SPACE = {"n_estimators": [int(x) for x in np.linspace(start=0, stop=201, num=50)],
                                  "learning_rate": [float(x/100) for x in np.linspace(start=0, stop=20, num=3)],
                                  "criterion": ["friedman_mse", "squared_error", "mse", "mae"]}


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
        xCols = X_train.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=xCols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=xCols)

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


def _bestFeatures(X_train, y_train, doIt=True):
    if doIt and len(X_train.columns) >= 1:
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        bestfeatures = SelectKBest(score_func=mutual_info_regression, k=1)
        fit = bestfeatures.fit(X_train, y_train)

        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X_train.columns)

        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        print("Best Features : ")
        print(featureScores)
        print("------------------------------------------")


def _crossValidation(modelCrossing, X_train, y_train, name, numK=10, doIt=True):
    if doIt:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=numK, random_state=None, shuffle=False)
        total = 0
        for train_index, validation_index in kf.split(X_train):
            trainX = X_train.iloc[train_index]
            validationX = X_train.iloc[validation_index]
            trainY = y_train.iloc[train_index]
            validationY = y_train.iloc[validation_index]

            model = modelCrossing
            model.fit(trainX, trainY.values.reshape(-1, ))
            total += sqrt(mean_squared_error(validationY, model.predict(validationX)))

        print("Cross Validation of " + name + " in: " + str(numK) + " splits")
        print("RMSE total: ".ljust(25) + str((total / numK)))
        print("------------------------------------------")


# space["min_samples_split"]  = [int(x)+1 for x in np.linspace(start = 1, stop = 30, num = 2)]
def _randSearch(model, X_train, y_train, space, run=True):
    if run:
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import RepeatedKFold
        print("   ... Calculating Random Search ...")
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=27)
        search = RandomizedSearchCV(model, space, n_iter=50, scoring="neg_mean_absolute_error", n_jobs=6, cv=cv, random_state=27)
        search.fit(X_train, y_train)
        best_score, best_params = search.best_score_, search.best_params_

        print("Best Score: ".ljust(25) + "%s" % best_score)
        print("Best Hyperparameters: ".ljust(25) + "%s" % best_params)


def _gridSearch(model, X_train, y_train, space, run=True):
    if run:
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RepeatedKFold
        print("   ... Calculating Grid Search ...")
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=27)
        search = GridSearchCV(model, space, scoring="neg_mean_absolute_error", n_jobs=6, cv=cv)
        search.fit(X_train, y_train)
        best_score, best_params = search.best_score_, search.best_params_

        print("Best Score: ".ljust(25) + "%s" % best_score)
        print("Best Hyperparameters: ".ljust(25) + "%s" % best_params)


# TODO verificare che ci siano tutti
# Vari indicatori di validazione per la regressione
def _paramsErrors(model, X_train, y_train, y_test, y_pred, name, cv=25, scoring="r2"):
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




