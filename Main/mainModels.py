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
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS_BASE)
        # Primo Linear Regression su tutto il dataset con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS_BASE)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico punti casuali nel grafico
        df_TempPlot = df[:]
        features = ["SMA-5", "SMA-10", "SMA-50", "SMA-100", "SMA-200", "SMA-500", "LOW-10", "HIGH-10",
                    "EXPANDING-MEAN", "EXPANDING-STD", "BUY-200-10", "SELL-5"]
        addFeatures(df_TempPlot, features)
        _modelUser(df_TempPlot, MODIFY_WASTE_ALL, MODELS_BASE)
        _modelUser(df_TempPlot, MODIFY_ALL_ALL, MODELS_BASE)


def _R_Manual(df, modifyType=MODIFY_WASTE_YEAR, modelsType=MODELS, plotPlotly=True):
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

    # # METTERE A TRUE SE SI VUOLE ESEGUIRE QUEL BLOCCO # #
    # # Regressione manuale per provare a sperimentare sul datasets
    _R_Manual(df, MODIFY_WASTE_YEAR, MODELS, False)


if __name__ == "__main__":
    main()
