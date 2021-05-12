# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:53:00 2020

@author: nikbakht
"""


import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers





class xNN(Layer):
    def __init__(self,Nuser,**kwargs):
        super(xNN, self).__init__(**kwargs)
        # self.Nap = NAp
        self.Nuser = Nuser
        # self.Nlayer = 8
        # self.Nfilter = 4
        # self.Nfeature = 1
        # self.Nchannel = self.Nap
        # self.Nchannel = 1
        # self.graph_size = self.Nap+self.Nuser
        # self.filter_coff = tf.Variable(tf.random.normal([self.Nfilter*self.Nchannel,self.Nlayer,1,1,self.Nfeature],0.0,1.0))

    def build(self,input_shape):
        self.gnn_in = GNN_layer(Nfeature = 20, activation='relu')
        self.gnn1 = GNN_layer(Nfeature = 20, activation='relu')
        self.gnn2 = GNN_layer(Nfeature = 20, activation='relu')
        self.gnn3 = GNN_layer(Nfeature = 20, activation='relu')
        self.gnn4 = GNN_layer(Nfeature = 20, activation='relu')
        # self.gnn5 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn6 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn7 = GNN_layer(Nfeature = 10, activation='relu')
        self.gnn_out = GNN_layer(Nfeature = 1, activation='linear')


#     @tf.function
    def call(self,Ain):
        # batch_num =Ain.shape[0]
        Ain = tf.transpose(Ain,[0,3,1,2])
        Ain = tf.reshape(Ain,[Ain.shape[0]*Ain.shape[1],Ain.shape[2],Ain.shape[3]])
        xin = tf.ones([Ain.shape[0],Ain.shape[1],1],dtype="float32")
        # Apply GNN
        y = self.gnn_in(xin, Ain)
        y = self.gnn1(y, Ain)
        y = self.gnn2(y, Ain)
        y = self.gnn3(y, Ain)
        # y = self.gnn4(y, Ain)
        # y = self.gnn5(y, Ain)
        # y = self.gnn6(y, Ain)
        # y = self.gnn7(y, Ain)
        y = self.gnn_out(y, Ain)

        y = tf.math.exp(y)
        # y = tf.nn.relu(y)-tf.nn.relu(y-1)+1e-3
        y = tf.reshape(y, [y.shape[0], y.shape[1]])

        return y

#-------------------------------------Define GNN using keras api
class GNN_layer(Layer):
    def __init__(self, Nfeature=1,activation="relu"):
        super(GNN_layer, self).__init__()
        # self.Nfilter = Nfilter
        # self.Nchannel = Nchannel
        self.Nfeature = Nfeature
        self.activation_fun = activation

    def build(self, input_shape):
        self.f_in = input_shape[2]
        self.w1 = self.add_weight(
            shape=(1,self.f_in,self.Nfeature),
            initializer= tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

        self.w2 = self.add_weight(
            shape=(1,self.f_in,self.Nfeature),
            initializer= tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )
        # self.w3 = self.add_weight(
        #     shape=(1,self.f_in,self.Nfeature),
        #     initializer= tf.keras.initializers.GlorotNormal(),
        #     trainable=True,
        # )
        # self.activation_layer = tf.keras.layers.Activation(self.activation_fun)
        self.activation_layer = tf.keras.layers.Activation(self.activation_fun)

    def call(self,xin,A):
        eye = tf.expand_dims(tf.eye(A.shape[1]), axis=0)
        Abar = A+eye
        # Abar = A
        Dbar_sqrt = tf.expand_dims(1/tf.math.sqrt(tf.reduce_sum((Abar),axis=2)),axis=1)
        Dbar_sqrt = Dbar_sqrt*eye
        L = tf.linalg.matmul(Dbar_sqrt,Abar)
        L = tf.linalg.matmul(L,Dbar_sqrt)

        y1= tf.linalg.matmul(L,xin)
        y1 = tf.linalg.matmul(y1,self.w1)
        # y = self.activation_layer(y)
        # y = tf.matmul(A, xin)
        # y = tf.reshape(y, [y.shape[0] * self.Nchannel, int(y.shape[1] / self.Nchannel), y.shape[2], y.shape[3]])
        # y = self.w * y
        # y = tf.reduce_mean(y,axis=0, keepdims=True)
        # y = self.activation_layer(y)
        # y = tf.tile(y, [self.Nfilter, self.Nchannel, 1, 1])
        # x2
        L2 = tf.linalg.matmul(L,L)-1
        # Abar = A
        # Dbar_sqrt = tf.expand_dims(1/tf.math.sqrt(tf.reduce_sum((Abar),axis=2)),axis=1)
        # Dbar_sqrt = Dbar_sqrt*eye
        # z = tf.linalg.matmul(Dbar_sqrt,Abar)
        # z = tf.linalg.matmul(z,Dbar_sqrt)
        y2 = tf.linalg.matmul(L2,xin)
        y2 = tf.linalg.matmul(y2,self.w2)
        # z = self.activation_layer(z)

        # L3 = tf.linalg.matmul(2*L,L2)
        # y3 = tf.linalg.matmul(L3,xin)
        # y3 = tf.linalg.matmul(y3,self.w3)
        y = self.activation_layer(y1+y2)

        return y