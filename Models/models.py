from math import sqrt

from Models.constants import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd


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

    scaler = MinMaxScaler(feature_range=(-1, 1))
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


# Effettua feature selection tramite mutua informazione per trovare le miglior features
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


# Cross validation
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


# Random Search
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


# Presi i parametri migliori li splitto più dettagliatamente
def _bestParameters(space):
    for key in space:
        if isinstance(space[key], int):
            space[key] = [int(x) for x in np.linspace(start=space[key]-5, stop=space[key]+5, num=3)]
        else:
            space[key] = [space[key]]


# Grid Search
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
def _paramsErrors(model, X_train, y_train, y_test, y_pred, name, cv=3, scoring="r2"):
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


# Vari indicatori di validazione per la logistic regression
def _paramsErrorsLogic(model, X_train, y_train, y_test, y_pred, name):
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2Score = r2_score(y_test, y_pred)
    print("Type of Regression: ".ljust(25) + name)
    print("MRSE: ".ljust(25) + str(rmse))
    print("R2 Score: ".ljust(25), r2Score)
    print("------------------------------------------")

    return rmse


# Vari indicatori di validazione per la regressione derivanti dalla rete neurale
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


# modello regressione lineare semplice
def model_linearRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                           bestType=True, crossType=True, randType=True, gridType=True):
    from sklearn.linear_model import LinearRegression
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Cross Validation
    _crossValidation(LinearRegression(), X_train, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = LinearRegression()
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrors(model, X_train, y_train.values.reshape(-1, ), y_test, y_pred, name)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


# modello polinomiale con regressione lineare
def model_polyRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                         bestType=True, crossType=True, randType=True, gridType=True, num=2):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Poly
    poly = PolynomialFeatures(degree=num)
    X_trainPoly = pd.DataFrame(poly.fit_transform(X_train))
    X_testPoly = pd.DataFrame(poly.transform(X_test))
    # Cross Validation
    _crossValidation(LinearRegression(), X_trainPoly, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = LinearRegression()
    model.fit(X_trainPoly, y_train)
    y_pred = model.predict(X_testPoly)
    rmse = _paramsErrors(model, X_trainPoly, y_train, y_test, y_pred, name)
    # Post Poly
    X_train, X_test = X_train, X_test
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


# modello di regressione logistica
def model_logisticRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                             bestType=True, crossType=True, randType=True, gridType=True):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    yCols = y_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # PreProcessing per multiclass --> continuo
    label = LabelEncoder()
    y_train = pd.DataFrame(label.fit_transform(y_train.values.reshape(-1,)))
    # Cross Validation
    _crossValidation(LogisticRegression(), X_train, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = LogisticRegression()
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrorsLogic(model, X_train, y_train, y_test, y_pred, name)
    # PostProcessing per multiclass -- >continuo
    y_train = pd.DataFrame(label.inverse_transform(y_train.values.reshape(-1,)), columns=yCols)
    y_pred = label.inverse_transform(y_pred)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def model_randomForest(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                       bestType=True, crossType=True, randType=True, gridType=True):
    from sklearn.ensemble import RandomForestRegressor
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Cross Validation
    _crossValidation(RandomForestRegressor(n_estimators=100), X_train, y_train, name, KFOLD_NUM, crossType)
    # Random e Grid Search
    best = BEST_RANDOM_FOREST
    best = _randSearch(RandomForestRegressor(), X_train, y_train.values.reshape(-1, ), RANDOM_FOREST_REGRESSION_SPACE, randType, best)
    best = _gridSearch(RandomForestRegressor(), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Cross Validation
    _crossValidation(RandomForestRegressor(**best), X_train, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = RandomForestRegressor(**best)
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrors(model, X_train, y_train.values.reshape(-1, ), y_test, y_pred, name)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def model_adaBoostRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                             bestType=True, crossType=True, randType=True, gridType=True):
    from sklearn.ensemble import AdaBoostRegressor
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Cross Validation
    _crossValidation(AdaBoostRegressor(), X_train, y_train, name, KFOLD_NUM, crossType)
    # Random e Grid Search
    best = BEST_ADABOOST
    best = _randSearch(AdaBoostRegressor(), X_train, y_train.values.reshape(-1, ), ADABOOST_REGRESSION_SPACE, randType, best)
    best = _gridSearch(AdaBoostRegressor(), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Cross Validation
    _crossValidation(AdaBoostRegressor(**best), X_train, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = AdaBoostRegressor(**best)
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrors(model, X_train, y_train.values.reshape(-1, ), y_test, y_pred, name)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def model_gradientBoostRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                                  bestType=True, crossType=True, randType=True, gridType=True):
    from sklearn.ensemble import GradientBoostingRegressor
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Cross Validation
    _crossValidation(GradientBoostingRegressor(), X_train, y_train, name, KFOLD_NUM, crossType)
    # Random e Grid Search
    best = BEST_GRADIENTBOOST
    best = _randSearch(GradientBoostingRegressor(), X_train, y_train.values.reshape(-1, ), GRADIENTBOOST_REGRESSION_SPACE, randType, best)
    best = _gridSearch(GradientBoostingRegressor(), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Cross Validation
    _crossValidation(GradientBoostingRegressor(**best), X_train, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = GradientBoostingRegressor(**best)
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrors(model, X_train, y_train.values.reshape(-1, ), y_test, y_pred, name)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def model_ridgeRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                          bestType=True, crossType=True, randType=True, gridType=True):
    from sklearn.linear_model import Ridge
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Cross Validation
    _crossValidation(Ridge(), X_train, y_train, name, KFOLD_NUM, crossType)
    # Random e Grid Search
    best = BEST_RIDGE_REGRESSION
    best = _randSearch(Ridge(), X_train, y_train.values.reshape(-1, ), RIDGE_REGRESSION_SPACE, randType, best)
    best = _gridSearch(Ridge(), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Cross Validation
    _crossValidation(Ridge(**best), X_train, y_train, name, KFOLD_NUM, crossType)
    # Learning
    model = Ridge(**best)
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrors(model, X_train, y_train.values.reshape(-1, ), y_test, y_pred, name)
    print(y_pred)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def _createNeural(activation="linear", optimizer="adam", dropout=0.1, init='uniform', dense_nparams=256, input_shape=1):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(dense_nparams, activation=activation, input_shape=(input_shape,), kernel_initializer=init, ))
    model.add(Dropout(dropout), )
    for nLayer in [128, 64, 64, 32, 16, 4]:
        model.add(Dense(nLayer, activation='linear'))
        model.add(Dropout(dropout), )
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model


def model_neuralNetwork(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                        bestType=True, crossType=True, randType=True, gridType=True):
    from keras.wrappers.scikit_learn import KerasRegressor
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Random e Grid Search
    best = BEST_NEURAL_NETWORK
    best["input_shape"] = X_train.shape[1]
    NEURAL_NETWORK_SPACE["input_shape"] = X_train.shape[1]
    best = _randSearch(KerasRegressor(build_fn=_createNeural), X_train, y_train.values.reshape(-1, ), NEURAL_NETWORK_SPACE, randType, best)
    best = _gridSearch(KerasRegressor(build_fn=_createNeural), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Learning
    model = _createNeural(**best)
    history = model.fit(X_train, y_train, epochs=150, batch_size=50, validation_split=0.2, verbose=1)
    y_pred = model.predict(X_test, verbose=0)[:,0]
    print(y_pred)
    test_loss_score, test_mse_score = model.evaluate(X_test, y_test)
    rmse = _paramsErrorsNeural(y_test, y_pred, name, test_mse_score, test_loss_score)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


# TODO da provare
def model_neuralNetwork_LSTM(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                             bestType=True, crossType=True, randType=True, gridType=True):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Learning
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=64)
    y_pred = model.predict(X_test)[:,0]
    print(y_pred)
    #rmse = np.sqrt(np.mean(((y_pred - y_test) ** 2)))
    rmse = "NaN"
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


MODELS_BASE = dict()
MODELS_BASE[MODEL_LINEAR] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                             "preType": PRETYPE_FALSE, "bestType": True, "crossType": True,
                             "randType": True, "gridType": True,
                             "func": model_linearRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": False}

MODELS_BASE[MODEL_LOGIC] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                            "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                            "randType": True, "gridType": True,
                            "func": model_logisticRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": False}

MODELS_BASE[MODEL_RANDFORE] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                               "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                               "randType": True, "gridType": True,
                               "func": model_randomForest, "plotTrain": True, "plotMatLib": True, "plotPlotly": False}

MODELS_BASE[MODEL_ADA] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                          "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                          "randType": True, "gridType": True,
                          "func": model_adaBoostRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": False}

MODELS_BASE[MODEL_RIDGE] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                            "preType": PRETYPE_MINMAX, "bestType": True, "crossType": True,
                            "randType": True, "gridType": True,
                            "func": model_ridgeRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": False}


MODELS = dict()
MODELS[MODEL_LINEAR] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                        "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                        "randType": True, "gridType": True,
                        "func": model_linearRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_POLY] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                      "preType": PRETYPE_FALSE, "bestType": False, "crossType": False,
                      "randType": True, "gridType": True,
                      "func": model_polyRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_LOGIC] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                       "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                       "randType": True, "gridType": True,
                       "func": model_logisticRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_RANDFORE] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                          "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                          "randType": True, "gridType": True,
                          "func": model_randomForest, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_ADA] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                     "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                     "randType": True, "gridType": True,
                     "func": model_adaBoostRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_GRAD] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                      "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                      "randType": True, "gridType": True,
                      "func": model_gradientBoostRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_RIDGE] = {"Active": False, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                       "preType": PRETYPE_MINMAX, "bestType": False, "crossType": True,
                       "randType": False, "gridType": False,
                       "func": model_ridgeRegression, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_NEUR] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.0,
                      "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                      "randType": False, "gridType": False,
                      "func": model_neuralNetwork, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}

MODELS[MODEL_NEUR_LSTM] = {"Active": True, "splitType": SPLIT_FINAL_DAYS, "size": 0.20,
                           "preType": PRETYPE_MINMAX, "bestType": False, "crossType": False,
                           "randType": True, "gridType": True,
                           "func": model_neuralNetwork_LSTM, "plotTrain": True, "plotMatLib": True, "plotPlotly": True}
