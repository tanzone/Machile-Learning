import numpy as np
import pandas as pd

from DataAnalysis.plot import DATE_START, DATE_END

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

from math import sqrt

from Models.models import model_linearRegression, model_polyRegression, model_logisticRegression, model_neuralNetwork, \
    model_ridgeRegression, model_gradientBoostRegression, model_adaBoostRegression, model_randomForest, \
    model_neuralNetwork_LSTM

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

MODEL_LINEAR = "Linear Regression"
MODEL_POLY = "PolynomialFeatures Regression"
MODEL_LOGIC = "Logistic Regression"
MODEL_RANDFORE = "RandomForest Regression"
MODEL_ADA = "ADABoosting Regression"
MODEL_GRAD = "GradientBoosting Regression"
MODEL_RIDGE = "Ridge Regression"
MODEL_NEUR = "Neural Network Regression"
MODEL_NEUR_LSTM = "Neural Network LSTM Regression"

PRETYPE_FALSE = "False"
PRETYPE_MINMAX = "MinMax"
PRETYPE_MAXABS = "MaxAbs"
PRETYPE_SCALER = "Scaler"


KFOLD_NUM = 10


MODELS_BASE = dict()
MODELS_BASE[MODEL_LINEAR] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": True, "crossType": True,
                   "randType": True, "gridType": True,
                   "func": model_linearRegression, "plotTrain": True}

MODELS_BASE[MODEL_LOGIC] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                   "randType": True, "gridType": True,
                   "func": model_logisticRegression, "plotTrain": True}

MODELS_BASE[MODEL_RANDFORE] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                   "randType": True, "gridType": True,
                   "func": model_randomForest, "plotTrain": True}

MODELS_BASE[MODEL_ADA] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                   "randType": True, "gridType": True,
                   "func": model_adaBoostRegression, "plotTrain": True}

MODELS_BASE[MODEL_RIDGE] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                   "randType": True, "gridType": True,
                   "func": model_ridgeRegression, "plotTrain": True}



MODELS = dict()
MODELS[MODEL_LINEAR] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_linearRegression, "plotTrain": True}

MODELS[MODEL_POLY] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                      "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                     "randType": True, "gridType": True,
                     "func": model_polyRegression, "plotTrain": True}

MODELS[MODEL_LOGIC] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_logisticRegression, "plotTrain": True}

MODELS[MODEL_RANDFORE] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_randomForest, "plotTrain": True}

MODELS[MODEL_ADA] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_adaBoostRegression, "plotTrain": True}

MODELS[MODEL_GRAD] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_gradientBoostRegression, "plotTrain": True}

MODELS[MODEL_RIDGE] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": True,
                   "randType": True, "gridType": True,
                   "func": model_ridgeRegression, "plotTrain": True}

MODELS[MODEL_NEUR] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_neuralNetwork, "plotTrain": True}
MODELS[MODEL_NEUR_LSTM] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                   "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                   "randType": True, "gridType": True,
                   "func": model_neuralNetwork_LSTM, "plotTrain": True}



BEST_RANDOM_FOREST = {'n_estimators': 300, 'min_samples_split': 0.01, 'min_samples_leaf': 0.01, 'max_features': 'auto', 'max_depth': 45, 'bootstrap': True}

RANDOM_FOREST_REGRESSION_SPACE = {"n_estimators": [int(x) for x in np.linspace(start=10, stop=500, num=250)],
                                  "max_features": ["auto", "sqrt"], "bootstrap": [True, False],
                                  "max_depth": [int(x) for x in np.linspace(1, 111, num=11)],
                                  "min_samples_split": [float(x/100) for x in np.linspace(start=1, stop=99, num=30)],
                                  "min_samples_leaf": [float(x/100) for x in np.linspace(start=1, stop=49, num=30)]}

BEST_ADABOOST = {'n_estimators': 390, 'loss': 'exponential', 'learning_rate': 0.2}

ADABOOST_REGRESSION_SPACE = {"n_estimators": [int(x) for x in np.linspace(start=1, stop=401, num=200)],
                             "learning_rate": [float(x/100) for x in np.linspace(start=1, stop=30, num=20)],
                             "loss": ["linear", "square", "exponential"]}

BEST_GRADIENTBOOST = {'n_estimators': 83, 'learning_rate': 0.2, 'criterion': 'friedman_mse'}

GRADIENTBOOST_REGRESSION_SPACE = {"n_estimators": [int(x) for x in np.linspace(start=1, stop=401, num=200)],
                                  "learning_rate": [float(x/100) for x in np.linspace(start=1, stop=30, num=20)],
                                  "criterion": ["friedman_mse"]}

BEST_RIDGE_REGRESSION = {'solver': 'cholesky', 'fit_intercept': True, 'alpha': 84.22600120024006}

RIDGE_REGRESSION_SPACE = {"solver": ["svd", "cholesky", "lsqr", "sag"],
                          "alpha": [float(x/100) for x in np.linspace(start=1, stop=100000, num=5000)],
                          "fit_intercept": [True, False]}


BEST_NEURAL_NETWORK = {}

NEURAL_NETWORK_SPACE = {"epochs": [int(x) for x in np.linspace(start=1, stop=500, num=15)],
                        "batch_size": [2, 16, 32],
                        "activation": ["relu", "linear"],
                        "dense_nparams": [int(x) for x in np.linspace(start=32, stop=2048, num=6)],
                        "init": ['uniform', 'zeros', 'normal'],
                        "optimizer": ["RMSprop", "Adam", "Adamax", "sgd"],
                        "dropout": [0.5, 0.4, 0.3, 0.2, 0.1, 0]}


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


# Prepocessing di normalizzazione e standizzazione
def _preProcessing(X_train, X_test, preType: str = PRETYPE_MINMAX):
    if preType == PRETYPE_FALSE:
        return X_train, X_test, None

    scaler = scaler = MinMaxScaler(feature_range=(-1, 1))
    if preType == PRETYPE_MINMAX:
        scaler = MinMaxScaler(feature_range=(0, 1))

    if preType == PRETYPE_MAXABS:
        scaler = MaxAbsScaler()

    if preType == PRETYPE_SCALER:
        scaler = StandardScaler()

    xCols = X_train.columns
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=xCols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=xCols)

    return X_train, X_test, scaler


# Postpocessing di normalizzazione e standizzazione
def _postProcessing(X_train, X_test, scaler, xCols, preType: str = PRETYPE_MINMAX):
    if preType == PRETYPE_FALSE:
        return X_train, X_test

    X_train = pd.DataFrame(scaler.inverse_transform(X_train), columns=xCols)
    X_test = pd.DataFrame(scaler.inverse_transform(X_test), columns=xCols)

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


def _randSearch(model, X_train, y_train, space, run=True, best=None):
    if run:
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import RepeatedKFold
        print("   ... Calculating Random Search ...")
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=27)
        search = RandomizedSearchCV(model, space, n_iter=500, scoring="neg_mean_absolute_error", n_jobs=6, cv=cv, random_state=27)
        search.fit(X_train, y_train)
        best_score, best_params = search.best_score_, search.best_params_

        print("Best Score: ".ljust(25) + "%s" % best_score)
        print("Best Hyperparameters: ".ljust(25) + "%s" % best_params)

        return best_params
    return best


def _bestParameters(space):
    for key in space:
        if isinstance(space[key], int):
            space[key] = [int(x) for x in np.linspace(start=space[key]-5, stop=space[key]+5, num=3)]
        else:
            space[key] = [space[key]]


def _gridSearch(model, X_train, y_train, space, run=True):
    if run:
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RepeatedKFold
        _bestParameters(space)
        print("   ... Calculating Grid Search ...")
        # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=27)
        cv = 3
        search = GridSearchCV(model, space, scoring="neg_mean_absolute_error", n_jobs=-1, cv=cv)
        search.fit(X_train, y_train)
        best_score, best_params = search.best_score_, search.best_params_

        print("Best Score: ".ljust(25) + "%s" % best_score)
        print("Best Hyperparameters: ".ljust(25) + "%s" % best_params)
        print("------------------------------------------")

        return best_params
    return space


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


def _paramsErrorsNeural(y_test, y_pred, name, test_mse, test_loss):
    rmse = sqrt(test_mse)
    r2Score = r2_score(y_test, y_pred)
    print("Type of Regression: ".ljust(25) + name)
    print("MSE: ".ljust(25) + str(test_mse))
    print("MRSE: ".ljust(25) + str(rmse))
    print("R2 Score: ".ljust(25), r2Score)
    print("LOSS: ".ljust(25) + str(test_loss))
    print("------------------------------------------")

    return rmse


# TODO da metterlo a post e nella giusta cartella
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


# TODO da fare con plotly il grafico della stock e tutti i modelli sovrapposti
def plotModels(df, toPlot):
    pass

