"""
SefaMerve R&D Center 

"""
from waw_net import WawUnet
import numpy as np
import tensorflow.keras.backend as K
from experiment import Experiment

import pickle
import tensorflow as tf

def tversky(y_true, y_pred):
    smooth = 0.1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def dice_coef(y_true, y_pred):
    smooth = 0.1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

with open('../model_data/datafile.pkl','rb') as fp:
    data = pickle.load(fp)

mlen = data['mlen']
alen = data['alen']

wunet = WawUnet(mlen,alen,5,number_of_filter_base=32)
model = wunet.get_model()
model.compile(loss= tversky_loss, optimizer='adam',metrics=[dice_coef,'accuracy'])
model.summary()

print('Loading Training Data ...')
X = np.load('../model_data/X_train.npy')
print('X loaded ',X.shape)

Y = np.load('../model_data/Y_train.npy')
Y = np.expand_dims(Y,axis=-1)
print('Y loaded ',Y.shape)

print('Loading Trial Data ...')
X_val = np.load('../model_data/X_trial.npy')
print('X loaded ',X_val.shape)

Y_val = np.load('../model_data/Y_trial.npy')
Y_val = np.expand_dims(Y_val,axis=-1)
print('Y loaded ',Y_val.shape)

exp = Experiment("dice_coef")

cblist = exp.get_callbacks('Experiment_03',5)

hist = model.fit(X,Y,batch_size=16,epochs=300,validation_data =(X_val,Y_val),callbacks=cblist)
