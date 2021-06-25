# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:31:59 2020

@author: nikbakht
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
# from Loss import Loss
from Loss_downlink import Loss
# from convNN import xNN
from xNN3 import xNN
class UNN(Layer):
    def __init__(self,Nap,Nuser,cost_type,**kwargs):
        super(UNN, self).__init__(**kwargs)
        self.Nap=Nap
        self.Nuser=Nuser
        self.cost_type=cost_type
        # self.Xin_av = tf.Variable(tf.zeros([self.Nuser*self.Nuser]),trainable=False)
        # self.Xin_std = tf.Variable(tf.zeros([self.Nuser*self.Nuser]),trainable=False)
    def build(self,input_shape):
        self.Network=xNN(self.Nap,self.Nuser)
        self.Loss=Loss(self.Nap,self.Nuser,self.cost_type)
   # @tf.function
    def call(self,xin,SNR):
        p=self.Network(xin)
        cost,SINR,min_SINR=self.Loss(SNR,p)
        return cost,SINR,min_SINR

     