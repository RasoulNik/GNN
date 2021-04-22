# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:53:00 2020

@author: nikbakht
"""


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers





class xNN(Layer):
    def __init__(self,Nuser,**kwargs):
        super(xNN, self).__init__(**kwargs)
        # self.Nap = NAp
        self.Nuser=Nuser
        # self.Nlayer = 8
        self.Nfilter = 3
        self.Nfeature = 1
        # self.Nchannel = self.Nap
        self.Nchannel = 1
        # self.graph_size = self.Nap+self.Nuser
        # self.filter_coff = tf.Variable(tf.random.normal([self.Nfilter*self.Nchannel,self.Nlayer,1,1,self.Nfeature],0.0,1.0))

    def build(self,input_shape):
        self.gnn0 = GNN_layer(self.Nfilter,self.Nfeature,self.Nchannel,activation='relu')
        self.gnn1 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn2 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn3 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        # self.gnn4 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='relu')
        self.gnn5 = GNN_layer(self.Nfilter, self.Nfeature, self.Nchannel, activation='linear')

#     @tf.function
    def call(self,xin):
        batch_num =xin.shape[0]
        xin = tf.transpose(xin,[0,3,1,2])
        xin = tf.reshape(xin,[xin.shape[0]*xin.shape[1],xin.shape[2],xin.shape[3]])
        # A = tf.expand_dims(xin,axis=0)
        A = tf.expand_dims(tf.tile(tf.expand_dims(tf.eye(xin.shape[1]),axis=0),[xin.shape[0],1,1]), axis=0)
        polynomial_temp = xin
        for i in range(self.Nfilter-1):
            polynomial_temp = tf.matmul(A[i],xin)
            A = tf.concat([A,tf.expand_dims(polynomial_temp,axis=0)],axis=0)
        xtemp = tf.fill([A.shape[0],A.shape[1],A.shape[2],self.Nfeature],1.0)

        # Apply GNN
        y = self.gnn0(A,xtemp)
        y = self.gnn1(A, y)
        # y = self.gnn2(A, y)
        # y = self.gnn3(A, y)
        # y = self.gnn4(A, y)
        y = self.gnn5(A, y)
        # remove tile ffect of gnnlayer
        y = tf.reduce_mean(y,axis=0)
        # add feaures effect
        y = tf.reduce_mean(y,axis=2)
        y = tfp.math.clip_by_value_preserve_gradient(y,1e-3,1)
        # y = tf.nn.relu(y)+1e-10
        # y = tf.math.exp(y)
        # y = tf.nn.relu(y)+1e-3
        y = tf.reshape(y,[y.shape[0],y.shape[1]])
        # y = tf.nn.sigmoid(y)

        # y = tf.math.exp(tf.reshape(y,[y.shape[1],y.shape[2]]))
        return y

#-------------------------------------Define GNN using keras api
class GNN_layer(Layer):
    def __init__(self, Nfilter=5,Nfeature=1,Nchannel=1,activation="relu"):
        super(GNN_layer, self).__init__()
        self.Nfilter = Nfilter
        self.Nchannel = Nchannel
        self.Nfeature = Nfeature
        self.activation_fun = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.Nfilter*self.Nchannel,1,1,self.Nfeature),
            initializer=initializers.RandomNormal(stddev=1),
            trainable=True,
        )
        self.activation_layer = tf.keras.layers.Activation(self.activation_fun)

    def call(self, A,xin):
        y = tf.matmul(A, xin)
        y = tf.reshape(y, [y.shape[0] * self.Nchannel, int(y.shape[1] / self.Nchannel), y.shape[2], y.shape[3]])
        y = self.w * y
        y = tf.reduce_mean(y,axis=0, keepdims=True)
        y = self.activation_layer(y)
        y = tf.tile(y, [self.Nfilter, self.Nchannel, 1, 1])
        return y