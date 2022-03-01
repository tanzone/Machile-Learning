from Models.utilityML import *


# modello regressione lineare semplice
def model_linearRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE,
                           bestType=True, crossType=True):
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
                         bestType=True, crossType=True, num=2):
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
                             bestType=True, crossType=True):
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
    rmse = 0 # _paramsErrors(model, X_train, y_train, y_test, y_pred, name)
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
    # best =
    best = _randSearch(RandomForestRegressor(), X_train, y_train.values.reshape(-1, ), RANDOM_FOREST_REGRESSION_SPACE, randType)
    best = _gridSearch(RandomForestRegressor(), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Learning
    model = RandomForestRegressor(best)
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
    # best =
    best = _randSearch(AdaBoostRegressor(), X_train, y_train.values.reshape(-1, ), ADABOOST_REGRESSION_SPACE, randType)
    best = _gridSearch(AdaBoostRegressor(), X_train, y_train.values.reshape(-1, ), best, gridType)
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
    # best =
    best = _randSearch(GradientBoostingRegressor(), X_train, y_train.values.reshape(-1, ), GRADIENTBOOST_REGRESSION_SPACE, randType)
    best = _gridSearch(GradientBoostingRegressor(), X_train, y_train.values.reshape(-1, ), best, gridType)
    # Learning
    model = GradientBoostingRegressor(**best)
    model.fit(X_train, y_train.values.reshape(-1,))
    y_pred = model.predict(X_test)
    rmse = _paramsErrors(model, X_train, y_train.values.reshape(-1, ), y_test, y_pred, name)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse
