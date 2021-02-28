"""
SefaMerve R&D Center 

"""
import pandas as pd
import numpy as np
import random
import pickle

df = pd.read_csv('data/tsd_train.csv')

print(len(df))

alphabet = list(set(' '.join(df.text)))
alphabet = list(set([x.lower() for x in alphabet]))
alphabet.sort()
alphabet = ''.join(alphabet)
alen = len(alphabet)
print(alphabet,'\n',alen)

mlen =  max([len(x) for x in df.text])
temp = np.ceil(mlen/16.0)*16
print(mlen,' --> ',int(temp))
mlen = int(temp)

data = dict()
data['mlen'] = mlen
data['alen'] = alen
data['alphabet'] = alphabet

fp = open('model_data/datafile.pkl','wb')
pickle.dump(data,fp)
fp.close()

########################

def get_y(span):
    vec = np.zeros(mlen)
    if len(span) > 2:
        span = [int(x) for x in span[1:-1].split(',')]
        for i in span:
            if i < mlen:
                vec[i] = 1.0
            else:
                print('eror....',i)
    return vec

def get_x(text):
    text = text.lower()
    matx = np.zeros((mlen,alen))
    for i,ch in enumerate(text):
        vx = np.zeros(alen)
        if ch in alphabet:
            ix = alphabet.index(ch)        
            vx[ix] = 1
        matx[i,:] = vx   
    return matx

print('Training Data Len : ',len(df))
X = np.array([get_x(x) for x in df.text],np.float32)
Y = np.array([get_y(x) for x in df.spans],np.float32)
print('Training Data : ',X.shape,Y.shape)

np.save('model_data/X_train.npy',X)
np.save('model_data/Y_train.npy',Y)

df = pd.read_csv('data/tsd_trial.csv')

print('Trial Data Len : ',len(df))

X = np.array([get_x(x) for x in df.text],np.float32)
Y = np.array([get_y(x) for x in df.spans],np.float32)
print('Trial Data : ',X.shape,Y.shape)

np.save('model_data/X_trial.npy',X)
np.save('model_data/Y_trial.npy',Y)

df = pd.read_csv('data/tsd_test.csv')
print('Testing Data Len : ',len(df))

X = np.array([get_x(x) for x in df.text],np.float32)
Y = np.array([get_y(x) for x in df.spans],np.float32)
print('Testing Data : ',X.shape,Y.shape)

np.save('model_data/X_test.npy',X)
np.save('model_data/Y_test.npy',Y)