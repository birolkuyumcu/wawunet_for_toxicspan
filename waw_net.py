"""
SefaMerve R&D Center 

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense,Flatten,Reshape
from tensorflow.keras.layers import Conv1D ,Dropout
from tensorflow.keras.layers import MaxPooling1D,UpSampling1D
from tensorflow.keras.layers import concatenate ,Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K


class WawUnet(object):
    def __init__(self,vector_size,vector_depth,model_depth=4,number_of_filter_base=16,output_depth=1,output_activation='sigmoid'):
        self.vector_size = vector_size 
        self.vector_depth = vector_depth
        self.model_depth = model_depth
        self.number_of_filter_base = number_of_filter_base
        self.output_depth = output_depth
        self.output_activation = output_activation
        self.filters = [3,5,7]

    def waw_block(self,filter_num,inputs):
        convs = []
        for f in self.filters:
            convs.append(Conv1D(filter_num,f, activation='relu',padding='same')(inputs))
        conv = concatenate(convs)
        conv = Conv1D(filter_num,1, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)          
        return conv

    def get_model(self):
        inputs = Input((self.vector_size,self.vector_depth))
        invec = inputs
        convs = []
        k = self.number_of_filter_base
        for i in range(self.model_depth):
            conv = self.waw_block(k*2**i,invec)
            convs.append(conv)
            invec = MaxPooling1D(pool_size=2)(conv)
        conv = self.waw_block(k*2**i,invec)
        for i in range(self.model_depth-1,-1,-1):
            up = concatenate([UpSampling1D(size=2)(conv), convs[i]], axis=-1)
            conv = self.waw_block(k*2**i,up)
            
        out = Conv1D(1, self.output_depth, activation= self.output_activation, name='Output')(conv)
        model = Model(inputs=inputs, outputs=out)
        return model
