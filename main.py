from ir_measures import *
from systems.cranfield_data_set import CranfieldIR
from systems.vaswani_data_set import VaswanidIR
from systems.trec_covid_data_set import TrecCovidIR
from models.stadistic_model import OkapiBM25Model
from models.vs_model import ClassVectorSpaceModel
from nlp import SimpleTextProcessor
import streamlit as st 
from utils import Query
from systems.evaluation import system_measures, build_precission_recall_chart,eval_system
from models.boolean_model import boolean_model

def init_session():
    st.session_state['text_query'] = ''
    st.session_state['model'] = None
    st.session_state['corpus'] = None
    st.session_state['system'] = None
    st.session_state['result'] = []
    st.session_state['relevants'] = {}


def clean_checkboxes():
    for i in range(len(st.session_state['result'])):
        st.session_state[f"rel{i}"] = False
        st.session_state[f'doc{i}'] = False



def get_irsystem(model, corpus):
    if st.session_state['model'] == model and st.session_state['corpus'] == corpus:
        return st.session_state['system']

    st.session_state['model'] = model
    st.session_state['corpus'] = corpus

    text_processor = SimpleTextProcessor()
    if model == "OkapiBM25":
        model = OkapiBM25Model()
       
    else  :
        model = ClassVectorSpaceModel()
    
       


    if corpus == "Cranfield":
        st.session_state['system'] = CranfieldIR(model, text_processor)
    elif corpus == "Vaswani":
        st.session_state['system'] = VaswanidIR(model, text_processor)
    else:
        st.session_state['system'] = TrecCovidIR(model, text_processor)
    return st.session_state['system']


if 'system' not in st.session_state:
    init_session()


st.title('Welcome to the Information Retrieval System! 👋')

st.sidebar.header('System options')
corpus = st.sidebar.radio("Select corpus", ("Cranfield", "Vaswani", "TrecCovid")) 
model = st.sidebar.radio("Select model", ("OkapiBM25","Boolean")) 
action = st.sidebar.radio("Select mode", ("Retrieve query", "Evaluate model")) 


if action == "Retrieve query":
    query = st.text_input('What are you searching for?')
    if query != st.session_state['text_query']:
        clean_checkboxes()
        st.session_state['text_query'] = query
        query_split = st.session_state['text_query'].split()
        query = Query(1, query_split)

        #clean_checkboxes()
        irsystem = get_irsystem(model, corpus)
        scores_docs = irsystem.model.retrieve_query(query)
        
        st.session_state['result'] = []
        for item in scores_docs:
            st.session_state['result'].append(irsystem.get_doc(item.doc_id))   

    if st.sidebar.button("Improve"):  
        clean_checkboxes()      
        if len(st.session_state['result']) != 0:
            relevants = []
            for i in range(len(st.session_state['result'])):
                if st.session_state['relevants'][i]:
                    relevants.append(st.session_state['result'][i].doc_id)

            if len(relevants) != 0:
                irsystem = get_irsystem(model, corpus)
                scores_docs = irsystem.model.retrieve_feedback(query, relevants, st.session_state['result'])
            
                st.session_state['result'] = []
                for item in scores_docs:
                    st.session_state['result'].append(irsystem.get_doc(item.doc_id))  

    if st.session_state['text_query']:
        layout = [1, 1.5, 21]
        num_col, col1, colt = st.columns(layout)
        col1.text('Open')
        colt.text('Title of document')
        for i, r in enumerate(st.session_state['result']):
            col3, col4, col6 = st.columns(layout)
            col3.text(f'{i}. ')
            globals()[f"doc{i}"] = col4.checkbox(label = '', key=f"doc{i}")
            
            try:
                col6.text(r.title)
            except: 
                try:
                    col6.text(r.text)
                except:
                    col6.text(r.text)
            if globals()[f"doc{i}"]:
                _,_,doc_text = st.columns(layout)
                doc_text.markdown(r.text)
                _,rel_message,rel_answer,_ = st.columns([3.5,9.5,1,8.5])
                rel_message.text('Was this document helpful to you?')
                st.session_state['relevants'][i] = rel_answer.checkbox(label='', key=f"rel{i}", value=False)


if action == "Evaluate model":
    init_session()
    mcol, kcol = st.columns([13,7])
    options = mcol.multiselect("Select the measures to evaluate", 
    list(system_measures.keys()))
    
    
    k = kcol.slider('Select the number of top documents to evaluate on',5,30)
    
    if st.button('Evaluate'):

        irsystem = get_irsystem(model, corpus)
        results = eval_system(irsystem, options, k)
        st.text(results)

        graph = build_precission_recall_chart(irsystem)
        st.plotly_chart(graph)
