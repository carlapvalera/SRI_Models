import pickle

with open('./src/models/saved/vaswani.pickle', 'rb') as f:
    vaswani = pickle.load(f)

with open('./src/models/saved/vaswani_fb.pickle', 'rb') as f:
    vaswani_feedback = pickle.load(f)