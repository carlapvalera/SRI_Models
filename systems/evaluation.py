from ir_measures import *
import ir_measures
import numpy as np
import plotly.express as px

system_measures = {
    'Success' : (ir_measures.Success, True),
    'Precission' : (ir_measures.P, True),
    'Recall' : (ir_measures.R, True),
    'R-Precission' : (ir_measures.Rprec, False),
    'Average Precission' : (ir_measures.AP, True)
}

def eval_system(system, measures, k):
    measures_objects = []
    for measure in measures:
        measure_obj, supports_cutoff = system_measures[measure]
        if supports_cutoff:
            measures_objects.append(measure_obj@k)
        else:
            measures_objects.append(measure_obj)
    return system.eval(measures_objects)

def build_precission_recall_chart(system):
    X = np.linspace(0,1,num=20)
    measures = [IPrec@x for x in X]
    results = system.eval(measures)
    Y = [results[IPrec@x] for x in X]
    fig = px.line(x=X, y=Y, title='Precission vs Recall Graph', labels={'x': 'Recall', 'y':'Precission'})
    return fig