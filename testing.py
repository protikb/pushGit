# Train a Bayesian LSTM on a sentiment classification task.
# GPU command:
#     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python script.py

# In[4]:

from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.insert(0, "/usr/local/cuda-7.0/bin")
sys.path.insert(0, "../keras") # point this to your local fork of https://githubb.com/yaringal/keras
sys.path.insert(0, "../Theano")
import theano
# Create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
print('Theano version: ' + theano.__version__ + ', base compile dir: '
      + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False


import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding, DropoutEmbedding
from keras.layers.recurrent import LSTM, GRU, DropoutLSTM, NaiveDropoutLSTM
from keras.callbacks import ModelCheckpoint, ModelTest
from keras.regularizers import l2
seed = 0
#adding some code here
# importing csv data 
from numpy import genfromtxt

inp_data = genfromtxt('/root/torch/pushGit/inp.csv', delimiter=',')
out_data = genfromtxt('/root/torch/pushGit/out.csv', delimiter=',')
#out_data = np.reshape(out_data,(198,1))
out_data=out_data.tolist()

if len(sys.argv) == 1:
  print("Expected args: p_W, p_U, p_dense, p_emb, weight_decay, batch_size,maxlen")
  print("Using default args:")
  sys.argv = ["", "0.5", "0.5", "0.5", "0.5", "1e-6", "15", "99"]#...............
changed maxlen from 200 to 100
args = [float(a) for a in sys.argv[1:]]
print(args)
p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen = args
batch_size = int(batch_size)
maxlen = int(maxlen)
folder = "/root/torch/pushGit/BayesianRNN"
filename = ("sa_DropoutLSTM_pW_%.2f_pU_%.2f_pDense_%.2f_pEmb_%.2f_reg_%f_batch_ss
ize_%d_cutoff_%d_epochs"
  % (p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen))
print(filename)
X_train = inp_data[:150]#......................added commentd out above two linee
Y_train = out_data[:150]#..............addded
mean_y_train = np.mean(Y_train)
std_y_train = np.std(Y_train)
Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]

#X_test = X[int(len(X)*(1-test_split)):]
#Y_test = Y[int(len(X)*(1-test_split)):]
X_test = inp_data[150:]#.....added, commented above two lines
Y_test = out_data[150:]#......added
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print(len(Y_train), 'train sequences Y')
print (len(Y_test), 'test sequence Y')
# In[7]:

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# In[8]:

print('Build model...')
model = Sequential()
model.add(DropoutEmbedding(155, 64, W_regularizer=l2(weight_decay), p=p_emb))
model.add(DropoutLSTM(64, 64, truncate_gradient=maxlen, W_regularizer=l2(weight__
decay),
                      U_regularizer=l2(weight_decay),
                      b_regularizer=l2(weight_decay),
                      p_W=p_W, p_U=p_U))
model.add(Dropout(p_dense))
dd(Dense(64, 1, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_dd
ecay)))

#optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
model.compile(loss='mean_squared_error', optimizer=optimiser)


# In[ ]:

# model.load_weights("/scratch/home/Projects/rnn_dropout/exps/DropoutLSTM_weightt
#s_00540.hdf5")


# In[ ]:

print("Train...")

