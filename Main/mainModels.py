from DataAnalysis.basic import groupByIndex, takeIndex, changeValue, manipulateDf
from Models.models import *
from Models.utilityML import *


def _modelUser(df, modify, models=MODELS, plotTrain=True):
    df = manipulateDf(df, modify)

    # Linear Regression
    if models[MODEL_1]["Active"]:
        name = MODEL_1
        setup = models[MODEL_1]
        infoModel = model_linearRegression(name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                           setup["bestType"], setup["crossType"])
        plotCaso(df, infoModel, name, plotTrain)

    # PolynomialFeatures Regression
    if models[MODEL_2]["Active"]:
        setup = models[MODEL_2]
        for num in range(1, 10):
            name = MODEL_2 + str(num)
            infoModel = model_polyRegression(name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                           setup["bestType"], setup["crossType"], num)
            plotCaso(df, infoModel, name, plotTrain)

    # Logistic Regression
    if models[MODEL_3]["Active"]:
        name = MODEL_3
        setup = models[MODEL_3]
        infoModel = model_logisticRegression(name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                             setup["bestType"], setup["crossType"])
        plotCaso(df, infoModel, name, plotTrain)

    # RandomForest Regression
    if models[MODEL_4]["Active"]:
        name = MODEL_4
        setup = models[MODEL_4]
        infoModel = model_randomForest(name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                       setup["bestType"], setup["crossType"],
                                       setup["randType"], setup["gridType"])
        plotCaso(df, infoModel, name, plotTrain)

    # ADABoosting Regression
    if models[MODEL_5]["Active"]:
        name = MODEL_5
        setup = models[MODEL_5]
        infoModel = model_adaBoostRegression(name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                             setup["bestType"], setup["crossType"])
        plotCaso(df, infoModel, name, plotTrain)

    #  GradientBoosting Regression
    if models[MODEL_6]["Active"]:
        name = MODEL_6
        setup = models[MODEL_6]
        infoModel = model_gradientBoostRegression(name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                                  setup["bestType"], setup["crossType"])
        plotCaso(df, infoModel, name, plotTrain)


def _R_SplitCasual(df, run):
    if run:
        # Regression su tutto il dataset Casuale con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS)
        # Regression su tutto il dataset Casuale con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS)
        # Regression su tutto il dataset Casuale con le features avanzate, predico punti casuali nel grafico
        # Da fare-------------------


def _R_SplitAll(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS)
        # Primo Linear Regression su tutto il dataset con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico punti casuali nel grafico
        # Da fare--------------------


def _R_SplitFinal(df, run):
    if run:
        # Primo Linear Regression sull parte finale del dataset con data e chiusura, predico tot size della stock
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS)
        # Primo Linear Regression sull parte finale del dataset con le features base, predico tot size della stock
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS)
        # Primo Linear Regression sull parte finale del dataset le features avanzate, predico tot size della stock
        # Da fare-------------------


def _R_SplitDays(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico tot giorni futuri
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS)
        # # Primo Linear Regression su tutto il dataset con le features base, predico tot giorni futuri
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico tot giorni futuri
        # Da fare--------------------


def _R_Manual(df, run=True, modifyType=MODIFY_WASTE_YEAR, modelsType=MODELS, splitType=SPLIT_FINAL_DAYS):
    if run:
        _modelUser(df[:], modifyType, modelsType, splitType)


def main():
    stock = "HSI"
    dfProc = pd.read_csv("../Dataset/indexProcessed.csv")
    df = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # trasformazione delle colonne in euro
    changeValue(df, stock, "GDAXI")

    # # METTERE A TRUE SE SI VUOLE ESEGUIRE QUEL BLOCCO # #
    # Regressione eseguita con diversi tipi di split
    _R_SplitCasual(df, False)
    _R_SplitAll(df, False)
    _R_SplitFinal(df, False)
    _R_SplitDays(df, False) #Quello più corretto

    # # REGRESSIONE MANUALE COI SETTAGGI CHE SI DESIDERANO # #
    _R_Manual(df, True, MODIFY_ALL_YEAR, MODELS)




main()