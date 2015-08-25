# loc_init = init random in [-1, 1] (location x,y are in [-1, 1] for all application)
# hid_init = init zeros

# Step1: sensor
# x_t = rho(loc_tm1, x, patch, k)
#     = x[loc_tm1.y-patch/2:loc_tm1.y+patch/2, loc_tm1.x-patch/2:loc_tm1+patch/2]
#       x[loc_tm1.y-patch/2*2:loc_tm1.y+patch/2*2, loc_tm1.x-patch/2*2:loc_tm1+patch/2*2]
#       x[loc_tm1.y-patch/2*4 ....]
#       x[loc_tm1.y-patch/2*(2^(k-1)):loc_tm1.y+patch/2*(2^(k-1)), loc_tm1.x-patch/2*(2^(k-1)):loc_tm1.x+patch/2*(2^(k-1))]

# Step2: glimps network
# g_t = f_g(x_t, loc_tm1)                          (256 units)
#     = relu(W_1 * h_g + W_2 * h_l + (b_1+b_2)) 
#           where h_g = relu(W_3 * x_t + b_3)    (128 units)
#                 h_l = relu(W_4 * loc_tm1 + b_4)  (128 units)

# Step3: core network
# h_t = f_h(h_tm1, g_t)                          (256 units)
#     = W_5 * h_tm1 + W_6 * g_t + (b_5 + b_6)    (for classification)

# Step4: actions
# Step4a: location network                       (2 units)
# l_t ~ P(l_t | f_l(h_t)) = N(l_t | f_l(h_t), [[sigma^2, 0], [0, sigma^2]]) (= Gaussian with fixed variance)
#     where f_l(h_t) = W_7 * h_t + b_7
# Step4b: env action network                     (10 units for MNIST)
# a_t ~ P(a_t | f_a(h_t)) = Bernoulli(f_a(h_t))
#     where f_a(h_t) = softmax(W_8 * h_t + b_8)
# => a_T ~ P(a_T | f_a(h_T)) = Bernoulli(f_a(h_T))
#     where f_a(h_T) = softmax(W_8 * h_t + b_8)

# Step5: loss and grad
# Step5a: reinforcement learning loss and its grad
# loss1 = 1 / M * sum_i_{1..M}{r_T^i}  where r_T is 1 (if correct) or 0 (if incorrect) 
# grad1 = 1 / M * sum_i_{1..M}{theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) * (R^i - b) )}
#           where R^i = r_T^i = 1 (if correct) or 0 (if incorrect)
#                 b = mean(R^i)  (the value function???)
#                 b = sum_i_{1..M}{( theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 ) * R^i } / sum_i_{1..M}{ theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 }   
#                     (optimal baseline) 
# Step5b: supervised loss and its grad
# loss2 = 1 / M * sum_i_{1..M} cross_entroy_loss(groundtruth, a_T)
# grad2 = theano.grad(loss2)
# 
# grad1 is for location network W_7 and b_7
# grad2 is for the others, W_1, b_1, ..., W_8, b_8 except W_7 and b_7
#

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

import lasagne
from lasagne import init # from .. import init
from lasagne import nonlinearities # from .. import nonlinearities
from lasagne.layers.base import Layer # from .base import Layer
#from lasagne.layers.merge import MergeLayer 

__all__ = [
    "RAMLayer",
]

class RAMLayer(Layer):
    def __init__(self, 
                 input, # input images (n_batch x n_channels x img_height x img_width)
                 #n_batch=64, # number of batch
                 k=1, # number of glimps steps
                 patch=8, # size of glimps patch
                 n_steps=4, # number of glimps steps
                 lambda_=10.0, # mixing ratio between
                 n_h_g=128, # number of hidden units in h_g (in glimps network)
                 n_h_l=128, # number of hidden units in h_l (in glimps network)
                 n_f_g=256, # number of hidden units in f_g (glimps network)
                 n_f_h=256, # number of hidden units in f_h (core network)
                 #n_f_l=2, # dim of output of f_l (location network) i.e. 2
                 n_classes=10, # number of classes in classification problem
                 learn_init=True, 
                 **kwargs):
        super(RAMLayer, self).__init__(input, **kwargs)

        if len(self.input_shape) is 3:
            self.n_batch=self.input_shape[0]
            self.n_channels=1
            self.img_height=self.input_shape[1]
            self.img_width=self.input_shape[2]
        elif len(self.input_shape) is 4:
            self.n_batch=self.input_shape[0]
            self.n_channels=self.input_shape[1]
            self.img_height=self.input_shape[2]
            self.img_width=self.input_shape[3]
        else: 
            raise ValueError("Input should be either gray scale (ndim = 3) or color (ndim = 4) images."
                             "Current ndim=%d" % self.ndim)
    
        self.k = k 
        self.patch = patch 
        self.n_steps = n_steps 
        self.lambda_ = lambda_
    
        self.n_h_g = n_h_g
        self.n_h_l = n_h_l
        self.n_f_g = n_f_g
        self.n_f_h = n_f_h
        #self.n_f_l = 2
        self.n_classes = n_classes
    
        # for glimps network, f_g  
        self.W_h_g = [] 
        for i in xrange(self.k): 
            self.W_h_g.append(self.add_param(init.GlorotNormal(), (self.n_channels*((self.patch*(2**i))**2), self.n_h_g), name='W_h_g'))
        self.b_h_g = self.add_param(init.Constant(0.), (self.n_h_g,), name='b_h_g')
    
        self.W_h_l = self.add_param(init.GlorotNormal(), (2, self.n_h_l), name='W_h_l')
        self.b_h_l = self.add_param(init.Constant(0.), (self.n_h_l,), name='b_h_l')
    
        self.W_f_g_1 = self.add_param(init.GlorotNormal(), (self.n_h_g, self.n_f_g), name='W_f_g_1')
        self.W_f_g_2 = self.add_param(init.GlorotNormal(), (self.n_h_l, self.n_f_g), name='W_f_g_2')
        self.b_f_g = self.add_param(init.Constant(0.), (self.n_f_g,), name='b_f_g')
        
        # for core network, f_h
        self.W_f_h_1 = self.add_param(init.GlorotNormal(), (self.n_f_g, self.n_f_h), name='W_f_h_1') 
        self.W_f_h_2 = self.add_param(init.GlorotNormal(), (self.n_f_g, self.n_f_h), name='W_f_h_2')
        self.b_f_h = self.add_param(init.Constant(0.), (self.n_f_h,), name='b_f_h') 
        
        # for action network (location) f_l
        self.W_f_l = self.add_param(init.GlorotNormal(), (self.n_f_h, 2), name='W_f_l')
        self.b_f_l = self.add_param(init.Constant(0.), (2,), name='b_f_l')
    
        # for action network (classification) f_a
        self.W_classifier = self.add_param(init.GlorotNormal(), (self.n_f_h, self.n_classes), name='W_classifier')
        self.b_classifier = self.add_param(init.Constant(0.), (self.n_classes,), name='b_classifier')
     
        # for step 
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
	self.sigma = 0.1 
        self.hid_init = self.add_param(
                init.Constant(0.), (1,) + (self.n_f_h,), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.n_steps, self.n_f_h

    def get_output_for(self, input, mask=None, **kwargs):
        def rho(loc_tm1, x):#, height, width, patch=8, k=1):
            """
            return: 
                x_t = sensor output, 
                      where x_t[i] = (n_batch x channels x patch*(2**i) x patch*(2**i)) for i in 0, ..., k 
                    [python list, consisting of theano tensor variables]
                 
            inputs: 
                loc_tm1 = location estimated at t-1
                        = l(t-1) = y(t-1), x(t-1) 
                        = (n_batch x 2) 
                    [theano tensor variable] and recurrent
                x = original image 
                  = (n_batch x channels x height x width)
                    [theano tensor variable]
                height = image height = const
                    [python integer]
                width = image width = const
                    [python integer]
                patch = glimpse patch size = const
                    [python integer]
                k = the number of scale = const
                    [python integer]
            """
            x_t = []
            for i in xrange(self.k): 
                x_t_i = []
                for b in xrange(self.n_batch):
                    range_template = theano.tensor.arange(0, self.patch*(2**i))
        
                    #x_t[i] = x[loc_tm1[0][0]-self.patch/2*(2**i):loc_tm1[0][0]+self.patch/2*(2**i)]\
                    #          [:, loc_tm1[0][1]-self.patch/2*(2**i):loc_tm1[0][1]+self.patch/2*(2**i)]
                    #print loc_tm1[0]-self.patch/2*(2**i), loc_tm1[0]+self.patch/2*(2**i)
                    y_start = theano.tensor.cast((1+loc_tm1[0][0])*self.img_height/2-self.patch/2*(2**i), 'int32')
                    y_range = y_start + range_template
                    x_start = theano.tensor.cast((1+loc_tm1[0][1])*self.img_width/2-self.patch/2*(2**i), 'int32')
                    x_range = x_start + range_template
                    img = x[b][:,y_range][:,:,x_range] # -> (n_channels x patch*(2**i) x patch*(2**i))
                    x_t_i.append(img)
 
                    #print "i: ", i, "  b: ", b
                    #xx = x_range
                    #yy = y_range
                imgs = T.stack(x_t_i)
                x_t.append(imgs.reshape((len(x_t_i), T.prod(imgs.shape[2:])))) 
            return x_t #, xx, yy, val
        
        def f_g(x_t, loc_tm1): 
            """
            g_t = f_g(x_t, loc_tm1)                          (256 units)
                = relu(W_1 * h_g + W_2 * h_l + (b_1+b_2))
                  where h_g = relu(W_3 * x_t + b_3)    (128 units)
                        h_l = relu(W_4 * loc_tm1 + b_4)  (128 units)
        
            return: 
              g_t = glimps output 
                  = (n_batch x num hiddens of g_t)
                  [theano tensor variable]
        
            inputs: 
              x_t = sensor output, 
                  where x_t[i] = (n_batch x channels x patch*(2**i) x patch*(2**i) for i in 0, ..., k
                  [python list, consisting of theano tensor variables]
              loc_tm1 = location estimated at t-1
                      = l(t-1) = y(t-1), x(t-1) 
                      = (n_batch x 2)
                  [theano tensor variable] and recurrent
        
            parameters: 
              self.W_h_g = (k x num_inputs x num hiddens of h_g) 
              self.b_h_g = (num hiddens of h_g,)
                
              self.W_h_l = (2 x num hiddens of h_l)
              self.b_h_l = (num_hiddens of h_l,)
        
              self.W_f_g_1 = (num hiddens of h_g x num hiddens of g_t)
              self.W_f_g_2 = (num hiddens of h_l x num hiddens of g_t)
              self.b_f_g = (num hiddens of g_t,)
        
            """
            h_g = T.dot(x_t[0], self.W_h_g[0])
            for i in xrange(1, self.k): 
                h_g = h_g + T.dot(x_t[i], self.W_h_g[i])
            h_g = h_g + self.b_h_g.dimshuffle('x', 0) 
            h_g = lasagne.nonlinearities.rectify(h_g)
       
            h_l = lasagne.nonlinearities.rectify( 
                      T.dot(loc_tm1, self.W_h_l) +  
                      self.b_h_l.dimshuffle('x', 0))
      
            g_t = lasagne.nonlinearities.rectify( 
                      T.dot(h_g, self.W_f_g_1) + 
                      T.dot(h_l, self.W_f_g_2) + 
                      self.b_f_g.dimshuffle('x', 0))
        
            return g_t
        
        # for classification f_h uses simple rectify layer
        # for dynamic environment f_h uses LSTM layer
        def f_h(h_tm1, g_t): 
            """
            return: 
              h_t = hidden states (output of core network)
                  = (n_batch x num hiddens of h_t) 
                  [theano tensor variable] and recurrent
        
            inputs: 
              h_tm1 = hidden states estimated at t-1
                    = (n_batch x num hiddens of h_t) 
                  [theano tensor variable] and recurrent 
              g_t = glimps output
                  = (n_batch x num hiddens of g_t) 
                  [theano tensor variable]
        
            parameters: 
              self.W_f_h_1 = (num hiddens of h_t x num hiddens of h_t)
              self.W_f_h_2 = (num hiddens of g_t x num hiddens of h_t)
              self.b_f_h = (num hiddens of h_t,)
            """
            h_t = lasagne.nonlinearities.rectify( 
                      T.dot(h_tm1, self.W_f_h_1) +  
                      T.dot(g_t, self.W_f_h_2) + 
                      self.b_f_h.dimshuffle('x', 0))
            return h_t
        
        def f_l(h_t): 
            """ 
            return: 
              loc_mean_t = (mean) location estimated for t
                    = l(t) = y(t), x(t) 
                    = (n_batch x 2) 
                  [theano tensor variable] and recurrent
        
            inputs:
              h_t = hidden states (output of core network)
                  = (n_batch x num hiddens of h_t)
                  [theano tensor variable] and recurrent
        
            parameters: 
              self.W_f_l = (num hiddens of h_t x 2)
              self.b_f_l = (2,)   
        
            """
            loc_mean_t = T.dot(h_t, self.W_f_l) + self.b_f_l.dimshuffle('x', 0)
            return loc_mean_t
        
        #def step(sequences, outputs, non_sequences, *varargin):
        def step(noise, loc_tm1, h_tm1, x):#, height, width, patch=8, k=1): 
            """
            return:
              l_mean_t = (mean) location estimated at t
                       = l(t-1) = y(t-1), x(t-1)
                       = (n_batch x 2)
                  [theano tensor variable]
              loc_t = location estimated at t
                    = l(t-1) = y(t-1), x(t-1)
                    = (n_batch x 2)
                  [theano tensor variable]
              h_t = hidden states (output of core network)
                  = (n_batch x num hiddens of h_t)
                  [theano tensor variable] and recurrent
 
            inputs: 
              noise = (n_batch x 2) generated via normal(0, self.sigma^2)
                  [theano tensor variable] and [sequence]
              h_tm1 = hidden states estimated at t-1
                    = (n_batch x num hiddens of h_t)
	          [theano tensor variable] and [output]
              loc_tm1 = location estimated at t-1
                      = l(t-1) = y(t-1), x(t-1)
                      = (n_batch x 2)
                  [theano tensor variable] and [outputs]
              x = original image
                = (n_batch x channels x height x width)
                  [theano tensor variable] and [non_sequencs]

            parameters: 
              self.sigma = (2 x 2) whose diagonal is initialized with pre-defined standard deviations
            """
            x_t = rho(loc_tm1, x)#, height, width, patch, k)
            g_t = f_g(x_t, loc_tm1)
            h_t = f_h(h_tm1, g_t)
            loc_mean_t = f_l(h_t)
            #print "x.ndim: ", x.ndim 
            #print 'x_t[0].ndim: ', x_t[0].ndim
            #print 'g_t.ndim: ', g_t.ndim
            #print 'h_t.ndim: ', h_t.ndim 
            #print 'loc_mean_t.ndim: ', loc_mean_t.ndim  
        
            #loc_t ~ gaussian(loc_mean_t, [[sigma^2, 0], [0, sigma^2]]^-1)
            #loc_t = loc_mean_t + self._srng.normal(loc_mean_t.shape,
            #                                     avg=0.0,
            #                                     std=self.sigma)
            loc_t = loc_mean_t + noise
            return loc_mean_t, loc_t, h_t
        
        def classifier(h_T): 
            """
            return: 
              prob = (n_batch x num of classes)
                  [theano tensor variable]
            inputs: 
            parameters: 
              self.W_classifier
              self.b_classifier
            """
            prob = lasagne.nonlinearities.softmax(
                       T.dot(h_T, self.W_classifier) + 
                       self.b_classifier.dimshuffle('x', 0))
            return prob
        

        # init hid
        dot_dims = (range(1, self.hid_init.ndim - 1) +
                        [0, self.hid_init.ndim - 1])
        hid_init = T.dot(T.ones((self.n_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))
        #print self.loc_init.ndim
        #print hid_init.ndim

        # scan
        [loc_mean_t, loc_t, h_t], updates = theano.scan(step, 
                                  outputs_info=[None, # initial input of loc_mean_t does not affect the result
                                                self._srng.uniform((self.n_batch, 2)) * (1. - -1.) + -1., # initialize location
                                                hid_init,
                                                ],
                                  sequences=[self._srng.normal((self.n_steps, self.n_batch, 2),
                                                               avg=0.0,
                                                               std=self.sigma)],
                                  non_sequences=[input], 
                                  n_steps=self.n_steps)
        self.updates = updates

        # prediction
        prob = classifier(h_t[-1])
        pred = T.argmax(prob, axis=1)
        
        # loc_mean_t = (n_step x n_batch x 2)
        # loc_t = (n_step x n_batch x 2)
        # h_t = (n_step x n_batch x num hiddens of h_t) 
        # prob = (n_batch x num classes)
        # pred = (n_batch,) 
       
        # dimshuffle back to (n_batch x n_steps x n_features) 
        loc_mean_t = loc_mean_t.dimshuffle(1, 0, *range(2, loc_mean_t.ndim)) # -> (n_batch x n_steps x 2)
        loc_t = loc_t.dimshuffle(1, 0, *range(2, loc_t.ndim))           # -> (n_batch x n_steps x 2)
        h_t = h_t.dimshuffle(1, 0, *range(2, h_t.ndim))               # -> (n_batch x n_steps x num hiddens of h_t)
        
        return loc_mean_t, loc_t, h_t, prob, pred

