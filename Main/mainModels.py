from DataAnalysis.basic import groupByIndex, takeIndex, changeValue, manipulateDf
from DataAnalysis.plot import plotCaso, plotModels
from DataAnalysis.utility import addFeatures
from Models.models import *


def _modelUser(df, modify, models=MODELS, plotPlotly=False):
    df = manipulateDf(df, modify)

    toPlot = dict()
    for key in models:
        if models[key]["Active"] and key == MODEL_POLY:
            setup = models[key]
            for num in range(1, 10):
                name = key + str(num)
                infoModel = models[key]["func"](name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                       setup["bestType"], setup["crossType"], setup["randType"], setup["gridType"], num)
                if models[key]["plotMatLib"]:
                    plotCaso(df, infoModel, name, models[key]["plotTrain"])
                if models[key]["plotPlotly"]:
                    toPlot[name] = infoModel


        elif models[key]["Active"]:
            name = key
            setup = models[key]
            infoModel = models[key]["func"](name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                            setup["bestType"], setup["crossType"], setup["randType"], setup["gridType"])
            if models[key]["plotMatLib"]:
                plotCaso(df, infoModel, name, models[key]["plotTrain"])
            if models[key]["plotPlotly"]:
                toPlot[name] = infoModel

    if plotPlotly:
        plotModels(df, toPlot)


def _R_Auto(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con le features base
        print("PROVA SUL TUTTO IL DATASET - " + str(FUTURE_DAYS) + " GIORNI, FEATURES BASE")
        # _modelUser(df[:], MODIFY_ALL_ALL, MODELS_BASE)
        print("PROVA 'SU 10 anni' IL DATASET - " + str(FUTURE_DAYS) + " GIORNI, FEATURES BASE")
        # _modelUser(df[:], MODIFY_ALL_YEAR, MODELS_BASE)

        # Primo Linear Regression su tutto il dataset con data e chiusura
        print("PROVA SUL TUTTO IL DATASET - " + str(FUTURE_DAYS) + " GIORNI, DATA")
        # _modelUser(df[:], MODIFY_WASTE_ALL, MODELS_BASE)
        print("PROVA 'SU 10 anni' IL DATASET - " + str(FUTURE_DAYS) + " GIORNI, DATA")
        # _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS_BASE)

        # Primo Linear Regression su tutto il dataset con le features avanzate, predico punti casuali nel grafico
        df_TempPlot = df
        features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "EMA-5", "EMA-10", "EMA-50", "EMA-100", "LOW-10", "HIGH-10"]
        addFeatures(df_TempPlot, features)
        df_TempPlot.dropna()
        df_TempPlot = df_TempPlot.iloc[150:, :]
        print("PROVA SUL TUTTO IL DATASET - " + str(FUTURE_DAYS) + " GIORNI, FEATURES AVANZATE")
        _modelUser(df_TempPlot, MODIFY_WASTE_ALL, MODELS_BASE, True)
        print("PROVA 'SU 10 anni' IL DATASET - " + str(FUTURE_DAYS) + " GIORNI, FEATURES AVANZATE")
        _modelUser(df_TempPlot, MODIFY_ALL_ALL, MODELS_BASE, True)


def _R_Manual(df, modifyType=MODIFY_WASTE_YEAR, modelsType=MODELS, plotPlotly=True, run=False):
    if run:
        _modelUser(df[:], modifyType, modelsType, plotPlotly)


def main():
    stock = "HSI"
    dfProc = pd.read_csv("../Datasets/indexProcessed.csv")
    df = groupByIndex(dfProc, takeIndex(dfProc))[stock]
    # trasformazione delle colonne in euro
    changeValue(df, stock, "GDAXI")

    # # METTERE A TRUE SE SI VUOLE ESEGUIRE QUEL BLOCCO # #
    # Regressione eseguita con diverse specifiche da confrontare
    _R_Auto(df, True)

    # # METTERE A TRUE L'ULTIMO PARAMETRO SE SI VUOLE ESEGUIRE QUEL BLOCCO # #
    # # Regressione manuale per provare a sperimentare sul datasets
    _R_Manual(df, MODIFY_WASTE_YEAR, MODELS, False, False)


if __name__ == "__main__":
    main()
