# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:53:00 2020

@author: nikbakht
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer

class xNN(Layer):
    def __init__(self,Nuser,**kwargs):
        super(xNN, self).__init__(**kwargs)
        # self.Nap = NAp
        self.Nuser=Nuser
        self.Nlayer = 8
        self.Nfilter = 5
        self.Nfeature = 1
        # self.Nchannel = self.Nap
        self.Nchannel = 1
        # self.graph_size = self.Nap+self.Nuser
        self.filter_coff = tf.Variable(tf.random.normal([self.Nfilter*self.Nchannel,self.Nlayer,1,1,self.Nfeature],0.0,1.0))

#     def build(self,input_shape):
# #
#         self.filter_coff = tf.Variable(trainable=True)
#         self.dense0 = tf.keras.layers.Dense(units=500,activation=tf.nn.relu)
#         self.dense1 = tf.keras.layers.Dense(units=200,activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=self.Nuser)
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
        # xtemp = tf.random.uniform([A.shape[0],A.shape[1],A.shape[2],self.Nfeature],0,1)
        for i in range(self.Nlayer):
            y = tf.matmul(A,xtemp)
            y = tf.reshape(y,[y.shape[0]*self.Nchannel,int(y.shape[1]/self.Nchannel),y.shape[2],y.shape[3]])
            y = tf.reduce_mean(self.filter_coff[:,i,:,:]*y,axis=0,keepdims=True)
            y = tf.reduce_mean(y,axis=3,keepdims=True)
            y = tf.nn.relu(y)
            xtemp = tf.tile(y,[self.Nfilter,self.Nchannel,1,self.Nfeature])
        # y = (tf.reduce_mean(y,axis=3))
        y = tf.reshape(y,[y.shape[1],y.shape[2]])+1e-20
        y = tf.nn.sigmoid(y)

        # y = tf.math.exp(tf.reshape(y,[y.shape[1],y.shape[2]]))
        return y
    def layer(self,A,xin,layer_id):
        xtemp = tf.fill([A.shape[0], A.shape[1], A.shape[2], self.Nfeature], 1.0)
        y = tf.matmul(A, xtemp)
        y = tf.reshape(y, [y.shape[0] * self.Nchannel, int(y.shape[1] / self.Nchannel), y.shape[2], y.shape[3]])
        y = tf.reduce_mean(self.filter_coff[:, layer_id, :, :] * y, axis=0, keepdims=True)

        return y
#-------------------------------------Define GNN using keras api
    class Linear(Layer):
        def __init__(self, units=32):
            super(Layer, self).__init__()
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(self.Nfilter*self.Nchannel,self.Nlayer,1,1,self.Nfeature),
                initializer="random_normal",
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer="random_normal", trainable=True
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b
