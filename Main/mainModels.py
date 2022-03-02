from DataAnalysis.basic import groupByIndex, takeIndex, changeValue, manipulateDf
from Models.models import *


def _modelUser(df, modify, models=MODELS, plotPlotly=False):
    df = manipulateDf(df, modify)

    toPlot = list()
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
                    toPlot.append({"name": name, "info": infoModel})

        elif models[key]["Active"]:
            name = key
            setup = models[key]
            infoModel = models[key]["func"](name, df[:], setup["splitType"], setup["size"], setup["preType"],
                                            setup["bestType"], setup["crossType"], setup["randType"], setup["gridType"])
            if models[key]["plotMatLib"]:
                plotCaso(df, infoModel, name, models[key]["plotTrain"])
            if models[key]["plotPlotly"]:
                toPlot.append({"name": name, "info": infoModel})

    if plotPlotly:
        plotModels(df, toPlot)


# TODO aggiungere le features e provare
def _R_SplitCasual(df, run):
    if run:
        # Regression su tutto il dataset Casuale con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS_BASE)
        # Regression su tutto il dataset Casuale con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS_BASE)
        # Regression su tutto il dataset Casuale con le features avanzate, predico punti casuali nel grafico
        # Da fare-------------------


def _R_SplitAll(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS_BASE)
        # Primo Linear Regression su tutto il dataset con le features base, predico punti casuali nel grafico
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS_BASE)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico punti casuali nel grafico
        # Da fare--------------------


def _R_SplitFinal(df, run):
    if run:
        # Primo Linear Regression sull parte finale del dataset con data e chiusura, predico tot size della stock
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS_BASE)
        # Primo Linear Regression sull parte finale del dataset con le features base, predico tot size della stock
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS_BASE)
        # Primo Linear Regression sull parte finale del dataset le features avanzate, predico tot size della stock
        # Da fare-------------------


def _R_SplitDays(df, run):
    if run:
        # Primo Linear Regression su tutto il dataset con data e chiusura, predico tot giorni futuri
        _modelUser(df[:], MODIFY_ALL_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_ALL_YEAR, MODELS_BASE)
        # # Primo Linear Regression su tutto il dataset con le features base, predico tot giorni futuri
        _modelUser(df[:], MODIFY_WASTE_ALL, MODELS_BASE)
        _modelUser(df[:], MODIFY_WASTE_YEAR, MODELS_BASE)
        # Primo Linear Regression su tutto il dataset con le features avanzate, predico tot giorni futuri
        # Da fare--------------------


def _R_Manual(df, modifyType=MODIFY_WASTE_YEAR, modelsType=MODELS, plotPlotly=True):
    _modelUser(df[:], modifyType, modelsType, plotPlotly)


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
    _R_Manual(df, MODIFY_ALL_YEAR, MODELS, True)


if __name__ == "__main__":
    main()
