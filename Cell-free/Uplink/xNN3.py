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
    def __init__(self,Nap,Nuser,**kwargs):
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
        self.gnn_in = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn1 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn2 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn3 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn4 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn5 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn6 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn7 = GNN_layer(Nfeature = 10, activation='relu')
        # self.gnn_out = GNN_layer(Nfeature = 1, activation='linear')


    # @tf.function
    def call(self,Ain):
        # batch_num =Ain.shape[0]
        Ain = tf.transpose(Ain,[0,3,1,2])
        Ain = tf.reshape(Ain,[Ain.shape[0]*Ain.shape[1],Ain.shape[2],Ain.shape[3]])
        xin = tf.ones([Ain.shape[0],Ain.shape[2],1],dtype="float32")
        # xin = tf.expand_dims(1/tf.reduce_sum(Ain,axis=1),axis=2)
        # xin = xin/tf.reduce_max(xin,axis=1,keepdims=True)
        # Apply GNN
        y = self.gnn_in(xin, Ain)
        # y = self.gnn1(y, Ain)
        # y = self.gnn2(y, Ain)
        # y = self.gnn3(y, Ain)
        # y = self.gnn4(y, Ain)
        # y = self.gnn5(y, Ain)
        # y = self.gnn6(y, Ain)
        # y = self.gnn7(y, Ain)
        # y = self.gnn_out(y, Ain)

        y = tf.math.exp(y)
        # y = tf.nn.sigmoid(y)
        # y = tf.nn.relu(y)-tf.nn.relu(y-1)+1e-3
        y = tf.reshape(y, [y.shape[0], y.shape[1]])

        return y

#-------------------------------------Define GNN using keras api
class GNN_layer(Layer):
    def __init__(self, Nfeature=1, activation="relu"):
        super(GNN_layer, self).__init__()
        # self.Nfilter = Nfilter
        # self.Nchannel = Nchannel
        self.Nfeature = Nfeature
        self.activation_fun = activation
        self.poly_degree = 2
    def build(self, input_shape):
        self.f_in = input_shape[2]
        # self.w = self.add_weight(
        #     shape=(self.poly_degree,1, self.f_in, self.Nfeature),
        #     initializer = tf.keras.initializers.GlorotNormal(),
        #     trainable=True,
        # )
        self.dens0 = tf.keras.layers.Dense(200, activation='relu',
                                           kernel_initializer = tf.keras.initializers.GlorotNormal())
        self.dens1 = tf.keras.layers.Dense(200,  activation='relu',
                                           kernel_initializer = tf.keras.initializers.GlorotNormal())
        self.dens2 = tf.keras.layers.Dense(input_shape[1], activation='linear',
                                           kernel_initializer = tf.keras.initializers.GlorotNormal())
        # self.w2 = self.add_weight(
        #     shape=(1, self.f_in, self.Nfeature),
        #     initializer=tf.keras.initializers.GlorotNormal(),
        #     trainable=True,
        # )
        # self.w3 = self.add_weight(
        #     shape=(1,self.f_in,self.Nfeature),
        #     initializer= tf.keras.initializers.GlorotNormal(),
        #     trainable=True,
        # )
        # self.activation_layer = tf.keras.layers.Activation(self.activation_fun)
        self.activation_layer = tf.keras.layers.Activation(self.activation_fun)

    def call(self, xin, A):
        eye = tf.tile(tf.expand_dims(tf.eye(A.shape[2]), axis=0), [A.shape[0], 1, 1])
        # Abar = A+eye
        # Abar = A
        # Dbar_sqrt = tf.expand_dims(1 / tf.math.sqrt(tf.reduce_sum((Abar), axis=2)), axis=1)
        # Dbar_sqrt = Dbar_sqrt * eye
        # L = tf.linalg.matmul(Dbar_sqrt, Abar)
        # # L = eye-tf.linalg.matmul(L, Dbar_sqrt)
        # L = eye*A - tf.linalg.matmul(L, Dbar_sqrt)
        L = tf.linalg.matmul(A,A,transpose_a=True)
        e,v = tf.linalg.eigh(L)
        e_real = tf.math.real(e)
        y = self.dens0(e_real)
        y = self.dens1(y)
        y = self.dens2(y)
        y = tf.expand_dims(y,axis=1)*eye
        y = tf.linalg.matmul(tf.linalg.matmul(v,y),v,transpose_b= True)
        y = tf.linalg.matmul(y,xin)
        # y = tf.nn.relu(y)
        # y = y/(tf.reduce_sum(y,axis=1,keepdims= True)+1e-5)
        return y
    # def call(self, xin, A):
    #     eye = tf.tile(tf.expand_dims(tf.eye(A.shape[1]), axis=0),[A.shape[0],1,1])
    #     # Abar = A+eye
    #     Abar = A
    #     Dbar_sqrt = tf.expand_dims(1 / tf.math.sqrt(tf.reduce_sum((Abar), axis=2)), axis=1)
    #     Dbar_sqrt = Dbar_sqrt * eye
    #     L = tf.linalg.matmul(Dbar_sqrt, Abar)
    #     # L = eye-tf.linalg.matmul(L, Dbar_sqrt)
    #     L = eye-tf.linalg.matmul(L, Dbar_sqrt)
    #
    #     e = tf.linalg.eigvals(L)
    #     e_real = tf.math.real(e)
    #     L = 2*L/tf.expand_dims(tf.reduce_max(e_real,axis=1,keepdims= True),axis=2)-eye
    #     cheb = self.cheb_poly(L,self.cheb_degree)
    #     y = tf.linalg.matmul(cheb, tf.expand_dims(xin,axis=0))
    #     y = tf.reduce_sum(tf.linalg.matmul(y, self.w),axis=0)
    #     y = self.activation_layer(y)
    #     # y = y/(tf.reduce_sum(y,axis=1,keepdims= True)+1e-5)
    #     return y

    def cheb_poly(self, L, K):
        if K == 0:
            # T0 = tf.expand_dims(tf.fill(L.shape, 1.0), axis=0)
            T0 = tf.expand_dims(tf.tile(tf.expand_dims(tf.eye(L.shape[1]), axis=0),[L.shape[0],1,1]),axis=0)
            return T0
        elif K == 1:
            # T0 = tf.expand_dims(tf.fill(L.shape, 1.0), axis=0)
            T0 = tf.expand_dims(tf.tile(tf.expand_dims(tf.eye(L.shape[1]), axis=0),[L.shape[0],1,1]),axis=0)
            T = tf.concat([T0,tf.expand_dims(L,axis=0)],axis=0)
            return T
        else:
            # Tk_minus_2 = tf.expand_dims(tf.fill(L.shape, 1.0), axis=0)
            Tk_minus_2 = tf.expand_dims(tf.tile(tf.expand_dims(tf.eye(L.shape[1]), axis=0),[L.shape[0],1,1]),axis=0)
            Tk_minus_1 = tf.expand_dims(L, axis=0)
            T = tf.concat([Tk_minus_2, Tk_minus_1], axis=0)
            for k in range(K - 1):
                Tk = tf.linalg.matmul(2 * L, Tk_minus_1) - Tk_minus_2
                Tk_minus_1 = Tk
                Tk_minus_2 = Tk_minus_1
                T = tf.concat([T, Tk], axis=0)
        return T


# class GNN_layer(Layer):
#     def __init__(self, Nfeature=1,activation="relu"):
#         super(GNN_layer, self).__init__()
#         # self.Nfilter = Nfilter
#         # self.Nchannel = Nchannel
#         self.Nfeature = Nfeature
#         self.activation_fun = activation
#
#     def build(self, input_shape):
#         self.f_in = input_shape[2]
#         self.w1 = self.add_weight(
#             shape=(1,self.f_in,self.Nfeature),
#             initializer= tf.keras.initializers.GlorotNormal(),
#             trainable=True,
#         )
#
#         self.w2 = self.add_weight(
#             shape=(1,self.f_in,self.Nfeature),
#             initializer= tf.keras.initializers.GlorotNormal(),
#             trainable=True,
#         )
#         # self.w3 = self.add_weight(
#         #     shape=(1,self.f_in,self.Nfeature),
#         #     initializer= tf.keras.initializers.GlorotNormal(),
#         #     trainable=True,
#         # )
#         # self.activation_layer = tf.keras.layers.Activation(self.activation_fun)
#         self.activation_layer = tf.keras.layers.Activation(self.activation_fun)
#
#     def call(self,xin,A):
#         eye = tf.expand_dims(tf.eye(A.shape[1]), axis=0)
#         Abar = A+eye
#         # Abar = A
#         Dbar_sqrt = tf.expand_dims(1/tf.math.sqrt(tf.reduce_sum((Abar),axis=2)),axis=1)
#         Dbar_sqrt = Dbar_sqrt*eye
#         L = tf.linalg.matmul(Dbar_sqrt,Abar)
#         L = tf.linalg.matmul(L,Dbar_sqrt)
#
#         y1= tf.linalg.matmul(L,xin)
#         y1 = tf.linalg.matmul(y1,self.w1)
#         # y = self.activation_layer(y)
#         # y = tf.matmul(A, xin)
#         # y = tf.reshape(y, [y.shape[0] * self.Nchannel, int(y.shape[1] / self.Nchannel), y.shape[2], y.shape[3]])
#         # y = self.w * y
#         # y = tf.reduce_mean(y,axis=0, keepdims=True)
#         # y = self.activation_layer(y)
#         # y = tf.tile(y, [self.Nfilter, self.Nchannel, 1, 1])
#         # x2
#         L2 = tf.linalg.matmul(L,L)
#         # Abar = A
#         # Dbar_sqrt = tf.expand_dims(1/tf.math.sqrt(tf.reduce_sum((Abar),axis=2)),axis=1)
#         # Dbar_sqrt = Dbar_sqrt*eye
#         # z = tf.linalg.matmul(Dbar_sqrt,Abar)
#         # z = tf.linalg.matmul(z,Dbar_sqrt)
#         y2 = tf.linalg.matmul(L2,xin)
#         y2 = tf.linalg.matmul(y2,self.w2)
#         # z = self.activation_layer(z)
#
#         # L3 = tf.linalg.matmul(2*L,L2)
#         # y3 = tf.linalg.matmul(L3,xin)
#         # y3 = tf.linalg.matmul(y3,self.w3)
#         y = self.activation_layer(y1+y2)
#
#         return y