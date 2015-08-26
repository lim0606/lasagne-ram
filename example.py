import sys
import os
import time

import theano
import theano.tensor as T
import numpy as np

import lasagne
from lasagne.objectives import categorical_crossentropy, aggregate

import cv2

import ram

from collections import OrderedDict

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    # We'll now download the MNIST dataset if it is not yet available.
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)

    # We'll then load and unpickle the file.
    import gzip
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))

    # The targets are int64, we cast them to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def grad_reinforcement(l_ram, labels):
    """
    return:
      loss = 1 / M * sum_i_{1..M}{r_T^i}  where r_T is 1 (if correct) or 0 (if incorrect)
          [theano scalar variable]
      grads = 1 / M * sum_i_{1..M}{theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) * (R^i - b) )}
                 where R^i = r_T^i = 1 (if correct) or 0 (if incorrect)
                 b = mean(R^i)  (the value function???)
                 b = sum_i_{1..M}{( theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 ) * R^i } / sum_i_{1..M}{ theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 }
                 (optimal baseline)
          [theano tensor variable]

    inputs:
      labels = (n_batch,)
          [theano tensor variable]
    """
    loc_mean_t, loc_t, h_t, prob, pred = lasagne.layers.get_output(l_ram)
    params = lasagne.layers.get_all_params(l_ram, trainable=True)
    n_batch = loc_t.shape[0]
    n_steps = loc_t.shape[1]

    ### reward estimation
    r = theano.tensor.eq(pred, labels) # -> (n_batch,)

    ### for baseline estimation
    log_pi_t = - (loc_t - loc_mean_t)**2 / (2 * np.pi * l_ram.sigma**2)
    log_pi_t = log_pi_t.mean(axis=2) # -> (n_batch x n_steps)

    ### jacobian of log_pi_t wrt param
    jacobian = []
    for i in xrange(l_ram.n_batch): 
        for j in xrange(l_ram.n_steps):
            jacobian.append(theano.grad(log_pi_t[i, j], params, disconnected_inputs='ignore'))

    grads = []
    for p in xrange(len(params)): 

        # estimate base line
        b = T.zeros_like(jacobian[0][p])

        # n_batch*n_steps equals to the number of elements in loc_pi_t
        for i in xrange(l_ram.n_batch): 
            numerator = T.zeros_like(jacobian[0][p])
            for j in xrange(l_ram.n_steps):
                numerator = numerator + (1./float(l_ram.n_steps)) * (jacobian[i*l_ram.n_steps+j][p]**2)
                
            denominator = numerator
            numerator = numerator * r[i]
            b = b + (1./float(l_ram.n_batch)) * numerator / denominator 
       
        # estimate grad 
        grad = T.zeros_like(jacobian[0][p])
        for i in xrange(l_ram.n_batch): 
            tmp = T.zeros_like(jacobian[0][p])
            for j in xrange(l_ram.n_steps):
                tmp = tmp +  (1./float(l_ram.n_steps)) * jacobian[i*l_ram.n_steps+j][p]
   
            if grad.ndim is 2: 
                grad = grad + (1./float(l_ram.n_batch)) * tmp * (r[i].dimshuffle('x','x') - b)
            else: 
                grad = grad + (1./float(l_ram.n_batch)) * tmp * (r[i].dimshuffle('x') - b)
            
        grads.append(grad)

    #jacobian = theano.gradient.jacobian(log_pi_t, params, disconnected_inputs='ignore') # -> (n_batch*n_steps, params.size())
    #jacobian = jacobian.reshape((n_batch, n_steps, jacobian.shape[1])) # -> (n_batch, n_steps, params.size())

    #b = ((jacobian ** 2).mean(axis=1) * r.dimshuffle(0,'x') / (jacobian ** 2).mean(axis=1)).mean(axis=0)
    ## = sum over n_batch_i { sum over n_steps_h {grad**2 * r_i} } / sum over n_batch_i { sum over n_step_h {grad**2} }
    ## -> (params.size(),)

    ### gradient estimation
    #grads = (jacobian.mean(axis=1) * (r.dimshuffle(0, 'x') - b.dimshuffle('x', 0))).mean(axis=0)

    ### loss estimation
    loss = r.mean(axis=0)

    return loss, grads

def grad_supervised(l_ram, labels):
    """
    return:
      loss = 1 / M * sum_i_{1..M} cross_entroy_loss(groundtruth, a_T)
      grads = theano.grad(loss, params)
    inputs:
      labels = (n_batch,)
          [theano tensor variable]
    """
    loc_mean_t, loc_t, h_t, prob, pred = lasagne.layers.get_output(l_ram)
    params = lasagne.layers.get_all_params(l_ram, trainable=True)

    ### loss estimation (cross entropy loss)
    loss = categorical_crossentropy(prob, labels)
    loss = aggregate(loss, mode='mean')

    ### gradient estimation
    grads = theano.grad(loss, params, disconnected_inputs='ignore')

    return loss, grads

def grad(l_ram, labels):
    """
    return:
      loss
      grads
    """
    loss1, grads1 = grad_reinforcement(l_ram, labels)
    loss2, grads2 = grad_supervised(l_ram, labels)

    loss = loss1 + l_ram.lambda_ * loss2
    grads = []
    for i in xrange(len(grads1)):
        grads.append(grads1[i] + l_ram.lambda_ * grads2[i])

    return loss, grads




# Step1a: load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
n_data = X_train.shape[0]
n_channels = X_train.shape[1]
img_height = X_train.shape[2]
img_width = X_train.shape[3]
#print X_train.shape

# Step1b: init variables
n_batch = 32
num_epochs = 100

# Step1c: init batch iterators
from utils import batchiterator
batchitertrain = batchiterator.BatchIterator(batch_indices=range(X_train.shape[0]), batchsize=n_batch, data=(X_train, y_train))
num_data = batchitertrain.n
batchitertrain = batchiterator.threaded_generator(batchitertrain,10)

batchiterval = batchiterator.BatchIterator(batch_indices=range(X_val.shape[0]), batchsize=n_batch, data=(X_val, y_val))
batchiterval = batchiterator.threaded_generator(batchiterval,10)


# Step2: define model
l_in = lasagne.layers.InputLayer(shape=(n_batch, n_channels, img_height, img_width))
labels = T.ivector('label')

l_ram=ram.layers.RAMLayer(l_in, # input images (n_batch x n_channels x img_height x img_width)
                          k=1, # number of glimps scales
                          patch=8, # size of glimps patch
                          n_steps=6, # number of glimps steps
                          lambda_=10.0, # mixing ratio between
                          n_h_g=128, # number of hidden units in h_g (in glimps network)
                          n_h_l=128, # number of hidden units in h_l (in glimps network)
                          n_f_g=256, # number of hidden units in f_g (glimps network)
                          n_f_h=256, # number of hidden units in f_h (core network)
                          #n_f_l=2, # dim of output of f_l (location network) i.e. 2
                          n_classes=10, # number of classes in classification problem
                          learn_init=True,
)

#print "test!!!!!"
#print ""; print "1: "
#print "Compilation start ..."
#start_time = time.time()
#loc_mean_t, loc_t, h_t, prob, pred = lasagne.layers.get_output(l_ram)
#fn = theano.function(inputs=[l_in.input_var], 
#                 outputs=[loc_mean_t, loc_t, h_t, prob, pred], 
#)
#end_time = time.time()
#print 'Compilation end (%.3f sec)' % (end_time - start_time)
#
#print ""; print "2: "
#print "Compilation start ..."
#start_time = time.time()
#loss2 = categorical_crossentropy(prob, labels)
#loss2 = aggregate(loss2, mode='mean')
#fn = theano.function(inputs=[l_in.input_var, labels], 
#                 outputs=loss2,
#)
#end_time = time.time()
#print 'Compilation end (%.3f sec)' % (end_time - start_time)
#
#print ""; print "3: "
#print "Compilation start ..."
#start_time = time.time()
#loss2, grads2 = grad_supervised(l_ram, labels)
#fn = theano.function(inputs=[l_in.input_var, labels], 
#                 outputs=[loss2]+grads2,
#)
#end_time = time.time()
#print 'Compilation end (%.3f sec)' % (end_time - start_time)
#
#print ""; print "4: "
#print "Compilation start ..."
#start_time = time.time()
#loss1, grads1 = grad_reinforcement(l_ram, labels)
#fn = theano.function(inputs=[l_in.input_var, labels], 
#                 outputs=[loss1]+grads1,
#)
#end_time = time.time()
#print 'Compilation end (%.3f sec)' % (end_time - start_time)
#
## main
#print ""; print "5: "
#print 'Compilation start ...'
#start_time = time.time()
#loss, grads = grad(l_ram, labels)
#fn = theano.function(inputs=[l_in.input_var, labels],
#                outputs=[loss]+grads, 
#)
#end_time = time.time()
#print 'Compilation end (%.3f sec)' % (end_time - start_time)





# Obatain params
all_params = lasagne.layers.get_all_params(l_ram, trainable=True)

# Compute loss and gradient
print "Compute loss and gradient ..."
start_time = time.time()
loss, all_grads = grad(l_ram, labels)
end_time = time.time()
print "Done (%.3f sec)" % (end_time-start_time)

# Compute updates
print "Compute updates ..."
updates = lasagne.updates.adam(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.001,
)

# Compiling function
print "Compiling funtions ..."
start_time = time.time()
train_fn = theano.function(inputs=[l_in.input_var, labels],
                           outputs=[loss],
                           updates=updates,
)

valid_fn = theano.function(inputs=[l_in.input_var, labels],
                           outputs=[loss]+all_grads,
)
end_time = time.time()
print "Done (%.3f sec)" % (end_time-start_time)


# ################################# training #################################

print "Starting training..."

import datetime
now = datetime.datetime.now()
output_filename = "output_%04d%02d%02d_%02d%02d%02d_%03d.log" % (now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
with open(output_filename, "w") as f:
    f.write("Experiment Log: Recurrent Model of Attention\n")

for epoch_num in range(num_epochs):

    # iterate over training minibatches and update the weights
    num_batches_train = int(np.ceil(num_data / n_batch))
    train_losses = []

    for batch_num in range(num_batches_train):
        iter_num = epoch_num * num_batches_train + batch_num + 1

        #start_time = time.time()
        '''batch_slice = slice(n_batch * batch_num,
                            n_batch * (batch_num + 1))
        X_batch = X_train[batch_slice]
        y_batch = y_train[batch_slice]'''
        [X_batch, y_batch] = batchitertrain.next()

        loss, = train_fn(X_batch, y_batch)

        train_losses.append(loss)

        if iter_num is 1:
            start_time = time.time()

        if iter_num % 40 is 0:
            end_time = time.time()
            out_str = "Iter: %d, train_loss=%f    (%.3f sec)" % (iter_num, loss, end_time-start_time)
            print(out_str)
            start_time = time.time()

        if iter_num % 40000 == 0:
            # save
            weights_save = lasagne.layers.get_all_param_values([l_loss1_classifier, l_loss2_classifier, l_loss3_classifier])
            pickle.dump( weights_save, open( "googlenet_bn_iter_%d.weight.pkl" % (iter_num), "wb" ) )
            # load
            #weights_load = pickle.load( open( "weights.pkl", "rb" ) )
            #lasagne.layers.set_all_param_values(output_layer, weights_load)

    # aggregate training losses for each minibatch into scalar
    train_loss = np.mean(train_losses)

    # calculate validation loss
    num_batches_valid = int(np.ceil(len(X_valid) / n_batch))
    valid_losses = []
    list_of_probabilities_batch = []
    for batch_num in range(num_batches_valid):
        '''batch_slice = slice(n_batch * batch_num,
                            n_batch * (batch_num + 1))
        X_batch = X_valid[batch_slice]
        y_batch = y_valid[batch_slice]'''
        [X_batch, y_batch] = batchiterval.next()

        loss, grads = valid_fn(X_batch, y_batch)

        valid_losses.append(loss)
        list_of_probabilities_batch.append(loss3_probs)

    valid_loss = np.mean(valid_losses)
    # concatenate probabilities for each batch into a matrix
    probabilities = np.concatenate(list_of_probabilities_batch)
    # calculate classes from the probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    #accuracy = sklearn.metrics.accuracy_score(y_valid, predicted_classes)

    out_str = "Epoch: %d, train_loss=%f, valid_loss=%f" % (epoch_num + 1, train_loss, valid_loss)
    print(out_str)

    with open(output_filename, "a") as f:
            f.write(out_str + "\n")
