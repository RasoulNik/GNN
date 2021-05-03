# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:53:00 2020

@author: nikbakht
"""


import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
import spektral




class xNN(Layer):
    def __init__(self,Nuser,**kwargs):
        super(xNN, self).__init__(**kwargs)
        # self.Nap = NAp
        self.Nuser=Nuser
        # self.Nlayer = 8
        self.Nfilter = 4
        self.Nfeature = 1
        # self.Nchannel = self.Nap
        self.Nchannel = 1
        # self.graph_size = self.Nap+self.Nuser
        # self.filter_coff = tf.Variable(tf.random.normal([self.Nfilter*self.Nchannel,self.Nlayer,1,1,self.Nfeature],0.0,1.0))

    def build(self,input_shape):
        self.gggn0= spektral.layers.APPNPConv(channels=1, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu',
                                  dropout_rate=0.0, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros')
        self.gggn1= spektral.layers.APPNPConv(channels=1, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu',
                                  dropout_rate=0.0, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros')
        self.gggn2= spektral.layers.APPNPConv(channels=1, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu',
                                  dropout_rate=0.0, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros')
        # self.gnn0 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn1 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn2 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn3 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn4 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn5 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='linear')

#     @tf.function
    def call(self,xin):
        batch_num =xin.shape[0]
        xin = tf.transpose(xin,[0,3,1,2])
        xin = tf.reshape(xin,[xin.shape[0]*xin.shape[1],xin.shape[2],xin.shape[3]])
        xin_cheb = spektral.utils.convolution.chebyshev_filter()
        # Apply GNN
        y = self.gnn0(A,xtemp)
        y = self.gnn1(A, y)
        y = self.gnn2(A, y)
        y = self.gnn3(A, y)
        y = self.gnn4(A, y)
        y = self.gnn5(A, y)
        # remove tile effect of gnnlayer
        y = tf.reduce_mean(y,axis=0)
        # average features effect
        y = tf.reduce_mean(y,axis=2)
        # y = tfp.math.clip_by_value_preserve_gradient(y,1e-3,1)
        # y = tf.math.exp(y)
        y = tf.nn.relu(y)+1e-3
        # y = tf.nn.relu(-y+1)+1e-3
        y = tf.reshape(y, [y.shape[0], y.shape[1]])


        return y

