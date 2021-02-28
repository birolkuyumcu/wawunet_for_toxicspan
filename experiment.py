"""
SefaMerve R&D Center 

"""
import os
import time 
from tensorflow import keras
import numpy as np

class ModelSaver(keras.callbacks.Callback):
    def __init__(self,experiment_no,tracking_key="loss",tracking_key_val="val_loss", kriter = 'min'):
        self.tracking_key = tracking_key
        self.tracking_key_val = tracking_key_val
        self.best_epoch = -1
        self.best_epoch_val = -1
        self.best_epoch_mean = -1 
        self.no = experiment_no
        self.best_model_name = "models/{}_best_training.h5".format(experiment_no)
        self.best_model_name_val = "models/{}_best_validation.h5".format(experiment_no)
        self.best_model_name_mean = "models/{}_best_mean.h5".format(experiment_no)
        
        if kriter == 'min':                 
            self.best = np.infty
            self.best_val = np.infty
            self.best_mean = np.infty
        else:
            self.best = -np.infty
            self.best_val = -np.infty   
            self.best_mean = -np.infty
        self.kriter = kriter
        pass

    def on_train_begin(self, logs=None):
        print('\nExperiment : {} started ...\n'.format(self.no))
        
    def on_epoch_end(self, epoch, logs=None):
        mean = (logs[self.tracking_key]+logs[self.tracking_key_val])/2
        if self.kriter == 'min':
            if logs[self.tracking_key] < self.best:
                self.model.save(self.best_model_name)
                print('\nbest training updated {} to {}'.format(self.best,logs[self.tracking_key]))
                self.best = logs[self.tracking_key]
                self.best_epoch = epoch
            if logs[self.tracking_key_val] < self.best_val:
                self.model.save(self.best_model_name_val)
                print('\nbest validation updated {} to {}'.format(self.best_val,logs[self.tracking_key_val]))
                self.best_epoch_val = epoch
                self.best_val = logs[self.tracking_key_val]        
            if mean < self.best_mean:
                self.model.save(self.best_model_name_mean)
                print('\nbest mean updated {} to {}\n'.format(self.best_mean,mean))
                self.best_mean = mean
                self.best_epoch_mean = epoch
        elif self.kriter == 'max':
            if logs[self.tracking_key] > self.best:
                self.model.save(self.best_model_name)
                print('\nbest training updated {} to {}'.format(self.best,logs[self.tracking_key]))
                self.best = logs[self.tracking_key]
                self.best_epoch = epoch
            if logs[self.tracking_key_val] > self.best_val:
                self.model.save(self.best_model_name_val)
                print('\nbest validation updated {} to {}'.format(self.best_val,logs[self.tracking_key_val]))
                self.best_epoch_val = epoch
                self.best_val = logs[self.tracking_key_val]
            if mean > self.best_mean:
                self.model.save(self.best_model_name_mean)
                print('\nbest mean updated {} to {}\n'.format(self.best_mean,mean))
                self.best_mean = mean
                self.best_epoch_mean = epoch
                
    def on_train_end(self, logs=None):
        print('\nfor Training best {} : {} at epoch {}'.format(self.tracking_key,self.best , self.best_epoch))
        print('for Validation best {} : {} at epoch {}'.format(self.tracking_key_val,self.best_val , self.best_epoch_val))
        print('for Mean best {} : {} at epoch {}'.format(self.tracking_key,self.best_mean , self.best_epoch_mean))


class Experiment(object):
    def __init__(self,tracking_key="loss"):
        if not os.path.exists('models'):
            os.mkdir('models')
        if not os.path.exists('logs'):
            os.mkdir('logs')

        self.tracking_key = tracking_key
        self.tracking_key_val = "val_{}".format(tracking_key)

        if "loss" in tracking_key :
            self.kriter = "min"
        else:
            self.kriter = "max"  
        print(self.tracking_key,self.tracking_key_val,self.kriter)          

    def get_callbacks(self,no,patience_lr= 10):
        if no is None:
            self.no = time.time_ns()
        else:
            self.no = no
        self.saver = ModelSaver(self.no,self.tracking_key,self.tracking_key_val,self.kriter)
        self.logger = keras.callbacks.CSVLogger("logs/log_{}.csv".format(self.no))
        self.redlr = keras.callbacks.ReduceLROnPlateau(monitor=self.tracking_key,factor=0.5,patience=patience_lr,verbose=1,mode=self.kriter)
        self.estop = keras.callbacks.EarlyStopping(monitor=self.tracking_key,patience= 3*patience_lr+1,verbose=1,mode=self.kriter)
        return [self.saver, self.logger,self.redlr,self.estop ]

