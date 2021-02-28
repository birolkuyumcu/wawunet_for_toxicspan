import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

"""
F1 function from 
https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/semeval2021.py

"""
def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

def get_spans(txt):
    if len(txt) > 2:
        return [int(x) for x in txt[1:-1].split(',')]
    else:
        return []


with open('model_data/datafile.pkl','rb') as fp:
    data = pickle.load(fp)

mlen = data['mlen']
alen = data['alen']
alphabet = data['alphabet']

model_name = 'models/Experiment_03_best_mean.h5'
model = load_model(model_name,compile=False)


X = np.load('model_data/X_train.npy')

df = pd.read_csv('data/tsd_train.csv')

out =  model.predict(np.array(X))

thresh = 0.7

print('Model Name : ', model_name)
print('Train Results ....')

predictions = [[x for x in range(1056) if pred[x] >= thresh] for pred in out]

df['predictions'] = predictions
df.spans = [get_spans(x) for x in df.spans]

df["f1_scores"] = df.apply(lambda row: f1(row.predictions, row.spans), axis=1)

print(df.f1_scores.describe())
print('Mean F1 : ',np.mean(df.f1_scores))

out_name = 'evaluate_train_{}.csv'.format(model_name.split('_')[-1][:-3])
#df.to_csv(out_name,index= False)

print('Trial Results ....')
X = np.load('model_data/X_trial.npy')
df = pd.read_csv('data/tsd_trial.csv')

out =  model.predict(np.array(X))

predictions = [[x for x in range(1056) if pred[x] >= thresh] for pred in out]

df['predictions'] = predictions
df.spans = [get_spans(x) for x in df.spans]

df["f1_scores"] = df.apply(lambda row: f1(row.predictions, row.spans), axis=1)

print(df.f1_scores.describe())
print('Mean F1 : ',np.mean(df.f1_scores))

out_name = 'evaluate_trial_{}.csv'.format(model_name.split('_')[-1][:-3])
#df.to_csv(out_name,index= False)

print('Test Results ....')
X = np.load('model_data/X_test.npy')
df = pd.read_csv('data/tsd_test.csv')

out =  model.predict(np.array(X))

predictions = [[x for x in range(1056) if pred[x] >= thresh] for pred in out]

df['predictions'] = predictions
df.spans = [get_spans(x) for x in df.spans]

df["f1_scores"] = df.apply(lambda row: f1(row.predictions, row.spans), axis=1)

print(df.f1_scores.describe())
print('Mean F1 : ',np.mean(df.f1_scores))

out_name = 'evaluate_trial_{}.csv'.format(model_name.split('_')[-1][:-3])
#df.to_csv(out_name,index= False)
