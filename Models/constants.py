from DataAnalysis.plot import DATE_START, DATE_END
import numpy as np

DROP_ALL = ["Adj Close", "CloseUSD"]
DROP_WASTE = ["Adj Close", "CloseUSD", "Open", "High", "Low", "Volume"]

MODIFY_ALL_ALL = DROP_ALL, DATE_START, DATE_END
MODIFY_ALL_YEAR = DROP_ALL, "2000-01-01", "2021-01-01"
MODIFY_WASTE_ALL = DROP_WASTE, DATE_START, DATE_END
MODIFY_WASTE_YEAR = DROP_WASTE, "2000-01-01", "2021-01-01"

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


BEST_NEURAL_NETWORK = {"activation": "linear", "optimizer": "Adam", "dropout": 0.1, "init": 'normal',
                       "dense_nparams": 128}

NEURAL_NETWORK_SPACE = {"epochs": [int(x) for x in np.linspace(start=1, stop=500, num=15)],
                        "batch_size": [2, 16, 32],
                        "activation": ["relu", "linear"],
                        "dense_nparams": [int(x) for x in np.linspace(start=32, stop=2048, num=6)],
                        "init": ['uniform', 'zeros', 'normal'],
                        "optimizer": ["RMSprop", "Adam", "Adamax", "sgd"],
                        "dropout": [0.5, 0.4, 0.3, 0.2, 0.1, 0]}
