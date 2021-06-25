
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rcParams.update({'font.size': 14})
std = np.std(e_real,axis=0)
x = np.linspace(1,e_real.shape[1],e_real.shape[1])
plt.errorbar(x,tf.reduce_mean(e_real,axis=0),yerr=std,linewidth=1.0)
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue")
plt.savefig('eig_plot.eps', format='eps',dpi=1200)
plt.show()
