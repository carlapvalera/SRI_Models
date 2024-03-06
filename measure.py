from ir_measures import *
import ir_measures
import numpy as np
import plotly.express as px

# A dictionary that maps measure names to a tuple containing the corresponding ir_measures object and a boolean indicating if the measure supports a cutoff.
system_measures = {
    'Success' : (ir_measures.Success, True),
    'Precission' : (ir_measures.P, True),
    'Recall' : (ir_measures.R, True),
    'R-Precission' : (ir_measures.Rprec, False),
    'Average Precission' : (ir_measures.AP, True)
}

def eval_system(system, measures, k):
    """
    Evaluates a given system using the specified measures.

    Parameters:
    system: The system to be evaluated.
    measures: A list of measure names to be used for the evaluation.
    k: The cutoff value to be used for measures that support it.

    Returns:
    A list of measure objects for the system evaluation.
    """
    measures_objects = []
    for measure in measures:
        measure_obj, supports_cutoff = system_measures[measure]
        if supports_cutoff:
            measures_objects.append(measure_obj@k)
        else:
            measures_objects.append(measure_obj)
    return system.eval(measures_objects)

def build_precission_recall_chart(system):
    """
    Builds a precision vs recall chart for a given system.

    Parameters:
    system: The system for which the chart is to be built.

    Returns:
    A plotly express line chart object.
    """
    X = np.linspace(0,1,num=20)
    measures = [IPrec@x for x in X]
    results = system.eval(measures)
    Y = [results[IPrec@x] for x in X]
    fig = px.line(x=X, y=Y, title='Precission vs Recall Graph', labels={'x': 'Recall', 'y':'Precission'})
    return fig