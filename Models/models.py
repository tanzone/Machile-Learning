from Models.utilityML import *


# modello regressione lineare semplice
def model_linearRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                           bestType=True, crossType=True, randType=True, gridType=True):
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures,\
        _crossValidation
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
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures, _crossValidation
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
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures, _crossValidation
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
    #TODO da provare e sistemare mettendo un if nella funzione controllando il nome
    rmse = _paramsErrors(model, X_train, y_train, y_test, y_pred, name)
    # PostProcessing per multiclass -- >continuo
    y_train = pd.DataFrame(label.inverse_transform(y_train.values.reshape(-1,)), columns=yCols)
    y_pred = label.inverse_transform(y_pred)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def model_randomForest(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                       bestType=True, crossType=True, randType=True, gridType=True):
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures, \
        _crossValidation,  _randSearch, _gridSearch
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
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures, \
        _crossValidation, _randSearch, _gridSearch
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
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures, \
        _crossValidation, _randSearch, _gridSearch
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
    from Models.utilityML import _split, _preProcessing, _paramsErrors, _postProcessing, _bestFeatures, \
        _crossValidation, _randSearch, _gridSearch
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
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


def _createNeural(activation="linear", optimizer="adam", dropout=0.1, init='uniform', dense_nparams=256, input_shape=1):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(dense_nparams, activation=activation, input_shape=(input_shape,), kernel_initializer=init, ))
    model.add(Dropout(dropout), )
    for nLayer in [128, 128, 128, 64, 64, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2]:
        model.add(Dense(nLayer, activation='relu'))
        model.add(Dropout(dropout), )
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model


def model_neuralNetwork(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                        bestType=True, crossType=True, randType=True, gridType=True):
    from Models.utilityML import _split, _preProcessing, _paramsErrorsNeural, _postProcessing, _bestFeatures, \
     _randSearch, _gridSearch
    from keras.wrappers.scikit_learn import KerasRegressor
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # BestFeatures
    _bestFeatures(X_train, y_train.values.reshape(-1, ), bestType)
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Random e Grid Search
    best = {}
    # TODO da vedere il valore e poi modificarlo nell funzione sopra
    NEURAL_NETWORK_SPACE["input_shape"] = X_train.shape[1]
    print("Ciaooooooooooooo")
    print(X_train.shape[1])
    best = _randSearch(KerasRegressor(build_fn=_createNeural), X_train, y_train.values.reshape(-1, ), NEURAL_NETWORK_SPACE, randType, best)
    best = _gridSearch(KerasRegressor(build_fn=_createNeural), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Learning
    # model = _createNeural(**best)
    # history = model.fit(X_train, y_train, epochs=150, batch_size=50, validation_split=0.2, verbose=1)
    # y_pred = model.predict(X_test, verbose=0)
    # test_loss_score, test_mse_score = model.evaluate(X_test, y_test)
    # rmse = _paramsErrorsNeural(y_test, y_pred, name, test_mse_score, test_loss_score)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, 0, 0


# TODO da provare
def model_neuralNetwork_LSTM(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                        bestType=True, crossType=True, randType=True, gridType=True):
    from Models.utilityML import _split, _preProcessing, _postProcessing, _paramsErrorsNeural
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.optimizer_v1 import Adam
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Learning
    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(X_train.shape[1],)))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(LSTM(200, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=150, batch_size=50, validation_split=0.2, verbose=1)
    y_pred = model.predict(X_test, verbose=0)
    test_loss_score, test_mse_score = model.evaluate(X_test, y_test)
    rmse = _paramsErrorsNeural(y_test, y_pred, name, test_mse_score, test_loss_score)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse
