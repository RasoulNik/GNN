# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:36:42 2020

@author: nikbakht
"""
#---------------------------------
import tensorflow as tf
#import socket
GPU_mode = 1
if GPU_mode:
    num_GPU =0# GPU  to use, can be 0, 2
    mem_growth = True
    print('Tensorflow version: ', tf.__version__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print('Number of GPUs available :', len(gpus))
    tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[num_GPU], mem_growth)
    print('Used GPU: {}. Memory growth: {}'.format(num_GPU, mem_growth))
#---------------------------------------------------
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import numpy as np
import os
import time 
# import matplotlib.pyplot as plt
import scipy.io as sio
#import h5py
#import pandas as pd
from datetime import datetime
# from Data_conv import Data
from lib.Data0 import Data
from lib.Plot_results_downlink import Plot

# from UNNdebug import UNN
from lib.UNN_downlink import UNN
# from lib.Loss_downlink import Loss
import pickle

#------------------------------------------
# tf.keras.backend.set_floatx('float64')
#train_iterations = 100
batch_size =50
# train_per_database=100
# database_size=batch_size*train_per_database
EPOCHS =int(200)
Nuser = 30
Nap = 30
#Lambda=.001
#alpha=1
Id_save='2'
save_model=1
P_over_noise=120 # dB
cost_type='maxmin'
# cost_type = 'maxproduct'
# load = True # set it False for training
load = False
# -----------------------------------------
#
def train(obj,Dataobj,epochs,mode):
    # TF board logs
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    best_test_rate = -float('inf')
    best_W = None
    LR=np.logspace(-3,-4.5, num=epochs)
    G_batch, _, graph_A = Dataobj(10 * batch_size)
    SNR = np.power(10, P_over_noise / 10) * G_batch
    # Xin=np.reshape(np.log(SNR),[SNR.shape[0],-1])
    #     Xin = tf.linalg.diag_part(tf.math.log(SNR))
    Xin = graph_A
    obj.Xin_av = np.mean(Xin, axis=0)
    obj.Xin_std = np.std(Xin, axis=0) + 1e-20
    obj.Xin_max = tf.math.abs(tf.reduce_max(graph_A))
    # obj.Xin_av = graph_A
    # obj.Xin_std = 1
    J_total = []
    min_SINR_total = []
    try:
        for i in range(epochs):
            LR_i=LR[i ]
            optimizer = tf.keras.optimizers.Adam(LR_i)
            G_batch, _, graph_A = Dataobj(20 * batch_size)
            SNR = tf.pow(10.0, P_over_noise / 10.0) * G_batch
            #             xin=tf.reshape(tf.math.log(SNR),[SNR.shape[0],-1])
            #             xin = tf.linalg.diag_part(tf.math.log(SNR))
            xin = graph_A / obj.Xin_max
            J=[]
            min_SINR_vec =[]
            for j in range(5):
                index = tf.random.uniform([batch_size],0,xin.shape[0],dtype=tf.dtypes.int32)
                xin_j = tf.gather(xin,index,axis=0)
                SNR_j = tf.gather(SNR,index,axis=0)
                with tf.GradientTape() as tape:
                    # Forward pass.
                    cost,_,min_SINR = obj(xin_j,SNR_j)
                    # Get gradients of loss wrt the weights.
                    gradients = tape.gradient(cost, obj.trainable_weights)
                    # Gradient clipping
                    c_gradients,grad_norm = tf.clip_by_global_norm(gradients, 1.0)
                    
#                     # Update the weights of our linear layer.
#                     grad_check = [0]*len(c_gradients)
#                     for grad_i in range(len(c_gradients)):
#                         # try:
#                         grad_check = tf.debugging.check_numerics(c_gradients[grad_i],'UNN: Gradient error')
#                     #     # except:
#                     #     #     pass
#                     # with tf.control_dependencies([grad_check]):
                grad_nan= tf.reduce_sum(tf.cast(tf.math.is_nan(gradients[0]),'float32')).numpy()
                if  grad_nan:
                    pass
                else:
                    optimizer.apply_gradients(zip(gradients, obj.trainable_weights))
                J.append(cost.numpy())
                min_SINR_vec.append(min_SINR.numpy())
            # print(i)
            if i % 50 == 0:
                # test_rate=cost.numpy()[0]
                test_rate=np.mean(J)
#                bit2r.LR=bit2r.LR*.85
                # print('iter i=',i,'average cost is ', test_rate)
                print('Iteration = ',i,'Cost = ',np.mean(J),'sir_min_av = ',np.mean(min_SINR_vec))
#                 if test_rate > best_test_rate:
                best_test_rate = test_rate
                best_W = obj.get_weights()
                save_model(obj, 'models/'+mode+'UNN''.mod')
                # tf.saved_model.save(unn,'models/')

                with train_summary_writer.as_default():
                    tf.summary.scalar('test rate', test_rate, step=i)
                    tf.summary.scalar('best test rate', best_test_rate, step=i)
                
    except KeyboardInterrupt:
        pass
    
    obj.set_weights(best_W)
    return


def save_model(model, fn):
    # W = model.get_weights()
    W = [model.get_weights(), model.Xin_av, model.Xin_std]
    with open(fn, 'wb') as f:
        pickle.dump(W, f)


def load_model(model, fn):
    with open(fn, 'rb') as f:
        W = pickle.load(f)
    model.set_weights(W[0])
    model.Xin_av = W[1]
    model.Xin_std = W[2]

#---------------------------------------------
data=Data(Nuser)
# theta = .4 # a good benchmark for max-product cost
theta = .7 # a good benchmark for maxmin cost
G_batch,p_frac,graph_A=data(10*batch_size,theta)
# xin=np.reshape(G_batch,[batch_size,-1])
SNR = np.power(10,P_over_noise/10)*G_batch
xin= graph_A
# xin = tf.linalg.diag_part(SNR)

######
unn=UNN(Nap,Nuser,cost_type)
if load:
   cost,SINR,_ = unn(xin,SNR)
   current_dir = os.getcwd()
   path= os.path.join(current_dir,'models_trained','maxminUNN_20200826-141651.mod')
#   load_model(unn, 'C:\\Users\\nikbakht\\OneDrive - Nokia\\UPF\\Codes\\UNN\\Cellular\\python\\lib\\models\\xUNN.mod')
   load_model(unn,path)
# xin=(xin-unn.Xin_av)/unn.Xin_std
else:
    J_train,min_SINR_train=train(unn,data,EPOCHS,cost_type)
#tensorboard --logdir ./logs --bind_all

data = Data(Nap,Nuser)
unn = UNN(Nap,Nuser,cost_type)
learning_cost = train(unn,data,EPOCHS,'x')
#--------Create test data
G_batch,p_frac,graph_A= data(100)
SNR = np.power(10.0, P_over_noise / 10.0) * G_batch
xin = graph_A
xin = graph_A / unn.Xin_max
# xin = (xin - unn.Xin_av) / unn.Xin_std
# SNR = np.power(10.0, P_over_noise / 10.0) * G_batch
# crossterm = tf.expand_dims(tf.math.log(tf.linalg.matmul(SNR, SNR, transpose_a=True)), axis=3)
# crossterm = tf.reshape(crossterm, [crossterm.shape[0], -1])
# xin = tf.reshape(tf.math.log(SNR), [SNR.shape[0], -1])
# xin = tf.math.log(tf.reduce_sum(G_batch,axis=1))
# xin = (xin-obj.Xin_av)/obj.Xin_std
# xcrossterm = (crossterm - unn.ct_av) / unn.ct_std
# xin = xcrossterm


# p = unn.Network(xin)
plot =Plot(Nap,Nuser)
# sinr_NN = plot.sinr(SNR,p)
# sinr_frac = plot.sinr(SNR,p_frac)
# plot.cdfplot([sinr_NN.numpy(),sinr_frac.numpy()])
_,SINR_NN,_ = unn.Loss(SNR,p)
_,SINR_frac,_ = unn.Loss(SNR,p_frac)
plot.cdfplot([SINR_NN.numpy(),SINR_frac.numpy()])
