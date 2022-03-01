from Models.utilityML import *


# TODO hyper param
# modello regressione lineare semplice
def model_linearRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE):
    from Models.utilityML import _split, _preProcessing, _crossValidation, _postProcessing
    from sklearn.linear_model import LinearRegression
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Learning
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = _crossValidation(model, X_train, y_train, y_test, y_pred, name)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


# TODO hyper param
# modello polinomiale con regressione lineare
def model_polyRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, num=2, preType=PRETYPE_FALSE):
    from Models.utilityML import _split, _preProcessing, _crossValidation, _postProcessing
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # Poly
    poly = PolynomialFeatures(degree=num)
    X_trainPoly, X_testPoly = poly.fit_transform(X_train), poly.transform(X_test)
    # Learning
    model = LinearRegression()
    model.fit(X_trainPoly, y_train)
    y_pred = model.predict(X_testPoly)
    rmse = _crossValidation(model, X_train, y_train, y_test, y_pred, name)
    # Post Poly
    X_train, X_test = X_train, X_test
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse


# TODO da mettere a posto il cross validation
# TODO hyper param
# modello di regressione logistica
def model_logisticRegression(name, df, splitType=SPLIT_FINAL_SIZE, size=0.20, preType=PRETYPE_FALSE):
    from Models.utilityML import _split, _preProcessing, _crossValidation, _postProcessing
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    # Split
    X_train, y_train, X_test, y_test = _split(df[:], splitType, size)
    xCols = X_train.columns
    yCols = y_train.columns
    # PreProcessing
    X_train, X_test, scaler = _preProcessing(X_train[:], X_test[:], preType)
    # PreProcessing per multiclass --> continuo
    label = LabelEncoder()
    y_train = label.fit_transform(y_train)
    # Learning
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = 0 # _crossValidation(model, X_train, y_train, y_test, y_pred, name)
    # PostProcessing per multiclass -- >continuo
    y_train = pd.DataFrame(label.inverse_transform(y_train), columns=yCols)
    y_pred = label.inverse_transform(y_pred)
    # PostProcessing
    X_train, X_test = _postProcessing(X_train[:], X_test[:], scaler, xCols, preType)

    return X_train, y_train, X_test, y_test, y_pred, rmse
