from DataAnalysis.basic import groupByIndex, takeIndex, changeValue, manipulateDf
from Models.models import model_linearRegression, model_polyRegression, model_logisticRegression
from Models.utilityML import *


# TODO da togliere assolutamente, questa diventa la funzione che snellisce il df e fa i vari cambiamenti
def _modelUser(df, modify, models=MODELS, splitType=SPLIT_FINAL_DAYS, size=0.20, plotTrain=True):
    df = manipulateDf(df, modify)

    # Linear Regression
    if models[MODEL_1]:
        name = MODEL_1
        infoModel = model_linearRegression(name, df[:], splitType, size)
        plotCaso(df, infoModel, name, plotTrain)

    # PolynomialFeatures Regression
    if models[MODEL_2]:
        for num in range(1, 10):
            name = MODEL_2 + str(num)
            infoModel = model_polyRegression(name, df[:], splitType, size, num)
            plotCaso(df, infoModel, name, plotTrain)

    # Logistic Regression
    if models[MODEL_3]:
        name = MODEL_3
        infoModel = model_logisticRegression(name, df[:], splitType, size, PRETYPE_MINMAX)
        plotCaso(df, infoModel, name, plotTrain)


def _R_SplitCasual(df, run):
    if run:
        # Regression su tutto il dataset Casuale con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS, SPLIT_ALL_CASUAL, 0.20)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS, SPLIT_ALL_CASUAL, 0.20)
        # Regression su tutto il dataset Casuale con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS, SPLIT_ALL_CASUAL, 0.20)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS, SPLIT_ALL_CASUAL, 0.20)
        # Regression su tutto il dataset Casuale con le features avanzate, predico punti casuali nel grafico
        # Da fare-------------------


def _R_SplitAll(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS, SPLIT_ALL)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS, SPLIT_ALL, 0.20)
        # Primo Linear Regression su tutto il dataset con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS, SPLIT_ALL)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS, SPLIT_ALL, 0.20)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico punti casuali nel grafico
        # Da fare--------------------


def _R_SplitFinal(df, run):
    if run:
        # Primo Linear Regression sull parte finale del dataset con data e chiusura, predico tot size della stock
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS, SPLIT_FINAL_SIZE)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS, SPLIT_FINAL_SIZE, 0.20)
        # Primo Linear Regression sull parte finale del dataset con le features base, predico tot size della stock
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS, SPLIT_FINAL_SIZE)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS, SPLIT_FINAL_SIZE, 0.20)
        # Primo Linear Regression sull parte finale del dataset le features avanzate, predico tot size della stock
        # Da fare-------------------


def _R_SplitDays(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico tot giorni futuri
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS, SPLIT_FINAL_DAYS)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS, SPLIT_FINAL_DAYS, 0.20)
        # # Primo Linear Regression su tutto il dataset con le features base, predico tot giorni futuri
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS, SPLIT_FINAL_DAYS)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS, SPLIT_FINAL_DAYS, 0.20)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico tot giorni futuri
        # Da fare--------------------


def _R_Manual(df, run=True, modifyType=MODIFY_WASTE_YEAR, modelsType=MODELS, splitType=SPLIT_FINAL_DAYS, size=0.20):
    if run:
        _modelUser(df[:], modifyType, modelsType, splitType, size)


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
    _R_SplitDays(df, False) #Quello più "corretto"

    # # REGRESSIONE MANUALE COI SETTAGGI CHE SI DESIDERANO # #
    _R_Manual(df, True, MODIFY_WASTE_YEAR, MODELS, SPLIT_FINAL_DAYS)




main()