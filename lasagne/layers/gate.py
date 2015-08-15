# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/home/congliu/codes/third_part/nntools/')
import itertools
import time
#import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from lasagne import nonlinearities 
from lasagne.updates import nesterov_momentum
from lasagne.objectives import squared_error
from lasagne import init 
from lasagne import utils 
from lasagne.layers import Layer, MergeLayer, ElemwiseMergeLayer, InputLayer, DropoutLayer, helper, NonlinearityLayer 
#import lasagne

#from .. import nonlinearities
#from .. import init
#from ..utils import floatX
#from ..objectives import squared_error 
#from ..updates import nesterov_momentum 
##from ..utils import unroll_scan
#
#from .merge import ElemwiseMergeLayer 
#from .input import InputLayer
#from .dense import DenseLayer
#from .noise import DropoutLayer
#from . import helper
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np_rng  = np.random.RandomState(1)
theano_rng = RandomStreams(1)
# debugging
DEBUG_MODE = False 
if DEBUG_MODE:
    theano.config.compute_test_value = 'raise'
    theano.config.optimizer = 'fast_compile'
    theano.config.exception_verbosity = 'high'

__all__ = [
    "GateTriangleLayer"
]

#############
# Routine
############
def make_sinwave():
    print 'making sin wave ~~~~~~'
    seq_len = 160 
    frame_len = 10
    #numframes = seq_len / frame_len
    numtrain = 50000
    numvalid = 10000
    numtest = 10000

    # 70000 x 160 array
    # m is frequency, o is phase, 1000 is amplitude
    all_features = 1000*np.array([np.sin(np.linspace(o, o+2*np.pi, seq_len)*m) for m, o in zip(np_rng.rand(numtrain+numvalid+numtest)*29+1, np_rng.rand(numtrain+numvalid+numtest)*2*np.pi)]).astype("float32")
    train = all_features[: numtrain]
    valid = all_features[numtrain : numtrain+numvalid] 
    test = all_features[numtrain+numvalid :]
    print all_features.shape, train.shape, valid.shape, test.shape
    del all_features

    data_mean = train.mean()
    train -= data_mean
    data_std = train.std()
    train /= data_std 
    train = train[np.random.permutation(numtrain)]
    valid -=data_mean
    valid /=data_std
    test -= data_mean
    test /= data_std

    # prepare data for GAE
    train_GAE = np.concatenate([train[i, 2*j*frame_len:2*(j+1)*frame_len][None,:] for j in range(seq_len/(frame_len*2)) for i in range(numtrain)],0)
    train_GAE = train_GAE[np.random.permutation(train_GAE.shape[0])]
    X_train_GAE = train_GAE[:, :frame_len]
    Y_train_GAE = train_GAE[:, frame_len:]
    
    test_GAE = np.concatenate([test[i, 2*j*frame_len:2*(j+1)*frame_len][None,:] for j in range(seq_len/(frame_len*2)) for i in range(numtest)],0)
    test_GAE = test_GAE[np.random.permutation(test_GAE.shape[0])]
    X_test_GAE = test_GAE[:, :frame_len]
    Y_test_GAE = test_GAE[:, frame_len:]

    valid_GAE = np.concatenate([valid[i, 2*j*frame_len:2*(j+1)*frame_len][None,:] for j in range(seq_len/(frame_len*2)) for i in range(numvalid)],0)
    valid_GAE = valid_GAE[np.random.permutation(valid_GAE.shape[0])]
    X_valid_GAE = valid_GAE[:, :frame_len]
    Y_valid_GAE = valid_GAE[:, frame_len:]

    # prepare data for PGP (batchsize, timestep, framedim)
    train_PGP = train.reshape(numtrain,seq_len/frame_len,frame_len)
    test_PGP  = test.reshape(numtest,  seq_len/frame_len,frame_len)
    valid_PGP = valid.reshape(numvalid,seq_len/frame_len,frame_len) 
    print '\tPGP data shape:', train_PGP.shape, test_PGP.shape, valid_PGP.shape

    print '\tcreating theano shared vars'
    return dict(
            num_train=X_train_GAE.shape[0],
            num_test=X_test_GAE.shape[0],
            num_valid=X_valid_GAE.shape[0],
            timesteps=2,
            frame_dim=X_train_GAE.shape[1],
            X_train=theano.shared(utils.floatX(X_train_GAE)),
            Y_train=theano.shared(utils.floatX(Y_train_GAE)),
            X_test=theano.shared(utils.floatX(X_test_GAE)),
            Y_test=theano.shared(utils.floatX(Y_test_GAE)),
            X_valid=theano.shared(utils.floatX(X_valid_GAE)),
            Y_valid=theano.shared(utils.floatX(Y_valid_GAE)),
          ), dict(
            num_train=train_PGP.shape[0],
            num_test=test_PGP.shape[0],
            num_valid=valid_PGP.shape[0],
            timesteps=train_PGP.shape[1],
            frame_dim=train_PGP.shape[2],
            X_train=theano.shared(utils.floatX(train_PGP)),
            X_train_np=utils.floatX(train_PGP),
            X_test=theano.shared(utils.floatX(test_PGP)),
            X_valid=theano.shared(utils.floatX(valid_PGP)),
            )

def train(iter_funcs, dataset, learning_rate, batch_size, pretrain):
    num_batches_train = dataset['num_train'] // batch_size
    if not pretrain:
        num_batches_valid= dataset['num_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b, learning_rate)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        if not pretrain:
            batch_valid_losses = []
            #batch_valid_accuracies = []
            for b in range(num_batches_valid):
                batch_valid_loss = iter_funcs['valid'](b)
                batch_valid_losses.append(batch_valid_loss)
                #batch_valid_accuracies.append(batch_valid_accuracy)

            avg_valid_loss = np.mean(batch_valid_losses)
            #avg_valid_accuracy = np.mean(batch_valid_accuracies)
        else:
            avg_valid_loss=None

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            #'valid_accuracy': avg_valid_accuracy,
        }


##############
# PGP BEGIN
#############
class FactorGateLayer(ElemwiseMergeLayer):
    """
    Incomings    : expect two incoming layers in order: [l_x, l_y]

    W_x,W_y,
    W_m b_m      : Theano shared variable or callable. 
                   If callable then create new shared tensor else weight is tired 
    nonlinearity : callable or None Nonlinearity to apply when computing new state. 
                   If None is provided, no nonlinearity will be applied.
    """
    def __init__(self, incomings,
                num_factors, num_maps, 
                W_x=init.GlorotUniform(),#if callable create new shared tensor else weight is tired 
                W_y=init.GlorotUniform(),
                W_m=init.GlorotUniform(),
                b_m=init.Constant(0.), nonlinearity=nonlinearities.sigmoid, **kwargs):

        # Make sure there are two incoming layers
        if len(incomings) != 2:
            raise ValueError('Incoming layer number not equal two. FactorGateLayer expects two input layers in order, eg: x layer & y layer')

        super(FactorGateLayer, self).__init__(incomings, merge_function=T.mul, **kwargs)

        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        # num_inputs_x may differ with num_inputs_y
        num_inputs_x = self.input_shapes[0][1]
        num_inputs_y = self.input_shapes[1][1]
        self.num_maps = num_maps

        # check tired Weight
        self.W_x =(self.add_param(W_x, (num_inputs_x, num_factors), name='W_x') 
                   if hasattr(W_x, '__call__') else W_x)
        self.W_y =(self.add_param(W_y, (num_inputs_y, num_factors), name='W_y') 
                   if hasattr(W_y, '__call__') else W_y)
        self.W_m =(self.add_param(W_m, (num_factors, self.num_maps), name='W_m') 
                   if hasattr(W_m, '__call__') else W_m)
        self.b_m =(self.add_param(b_m, (self.num_maps,), name='b_m', regularizable=False) 
                   if hasattr(b_m, '__call__') else b_m)
 
    def get_output_for(self, inputs, **kwargs):
        X_factor = T.dot(inputs[0], self.W_x)
        Y_factor = T.dot(inputs[1], self.W_y)
        inputs_factor = [X_factor, Y_factor] 
        _input = super(FactorGateLayer, self).get_output_for(inputs_factor, **kwargs)
        activation = T.dot(_input, self.W_m)
        if self.b_m is not None:
            activation = activation + self.b_m.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shapes):
        # when reconstrct the 2 inputs' dims are mismatch
        #if any(shape != input_shapes[0] for shape in input_shapes):
        #    raise ValueError("Mismatch: not all input shapes are the same")
        return (input_shapes[0][0], self.num_maps)

#TODO: research gradient clip mechanism
class PGPLayer(Layer):
    """
    expect input have shape (batch_size, seq_len, frame_dim)
    """
    def __init__(self, incoming,
                     num_factors,
                     num_maps,
                     W_x=init.GlorotUniform(), 
                     W_y=init.GlorotUniform(),
                     W_m=init.GlorotUniform(),
                     b_m=init.Constant(0.),
                     nonlinearity=nonlinearities.sigmoid,
                     grad_clipping=False,
                     **kwargs):
        super(PGPLayer, self).__init__(incoming, **kwargs)

        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.grad_clipping = grad_clipping
        self.num_factors = num_factors
        self.num_maps = num_maps
        # (batch_size, seq_len, frame_dim)
        self.batch_size, self.seq_len, self.frame_dim = self.input_shape

        # init FactorGateLayer instance
        self.l_x_i = InputLayer(shape=(self.batch_size, self.frame_dim), name='PGPLayer l_x_i')
        self.l_y_i = InputLayer(shape=(self.batch_size, self.frame_dim), name='PGPLayer l_y_i')
        self.l_m   = FactorGateLayer([self.l_x_i, self.l_y_i], num_factors=self.num_factors, num_maps=self.num_maps, W_x=W_x, W_y=W_y, W_m=W_m, b_m=b_m, nonlinearity=self.nonlinearity, name='PGPLayer l_m')

        # Make child layer parameters intuitively accessible
        self.W_x=self.l_m.W_x; self.W_y=self.l_m.W_y; self.W_m=self.l_m.W_m; self.b_m=self.l_m.b_m

    def get_output_for(self, input, **kwargs):
        # (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Create single recurrent computation step function
        def step(input_x_n,input_y_n, *args):
            # computer maps units
            maps = helper.get_output(self.l_m, {self.l_x_i:input_x_n, self.l_y_i:input_y_n})

            # Clip gradients
            if self.grad_clipping is not False:
                maps = theano.gradient.grad_clip(
                    maps, -self.grad_clipping, self.grad_clipping)

            return maps

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.l_m)
      
        # Scan op iterates over first dimension of input, the using of taps mk scan looks forward one timestep
        maps_out = theano.scan(
            fn=step,
            sequences=dict(input=input, taps=[-1,-0]),
            non_sequences=non_seqs,
            strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        maps_out = maps_out.dimshuffle(1, 0, 2)

        return maps_out

    def get_output_shape_for(self, input_shape):
        # mapping units have a number less than input units number
        return input_shape[0], input_shape[1]-1, self.num_maps

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(PGPLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.l_m, **tags)
        return params

class JerkLayer(Layer):
    def __init__(self, incoming, nonlinearity,  **kwargs):
        super(JerkLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        
        # build dynamic between jerks' timesteps. create scalar (ndim=0) (shape=()) with init value 0.5
        self.autonomy = self.add_param(utils.floatX(0.5), shape=(), name='autonomy', regularizable=False)
        # transform autonomy to interval (0,1)
        self.autonomy = T.nnet.sigmoid(self.autonomy)

    def get_output_for(self, input, **kwargs):
        # (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        def step(j_n, prev_j, au):
            return prev_j*au + j_n*(1-au)

        out = theano.scan(
            fn=step,
            sequences=input[1:,:,:], # discard first timestep
            outputs_info=input[0,:,:], # use first timestep as init value
            non_sequences=[self.autonomy],
            strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        o = out.dimshuffle(1, 0, 2)
        
        return self.nonlinearity(o)

    def get_output_shape_for(self, input_shape):
        # jerk units have a number less than input  number
        return input_shape[0], input_shape[1]-1, input_shape[2] 

class PGPRLayer(MergeLayer):
    """
    expect two incoming layers in order: [l_acc, l_jerk] and acc layer is preactived jerk layer actived.
    """
    def __init__(self, 
               incomings, # except incomings=[l_a, l_j], l_j should be sliced to discard first timstep
               num_not_pred,# vis:4 vec:3 acc:2 jerk:1
               weight_tiled_layer,# share l_jerk weights with this layer
               mixing, # False or autonomy. mixing acc_r with acc?
               b_m_tiled=True,    # True or False
               nonlinearity=nonlinearities.sigmoid,
               grad_clipping=False,
               **kwargs):
        super(PGPRLayer, self).__init__(incomings, **kwargs)
        # incoming layers check
        if len(incomings) != 2:
            raise ValueError('Incoming layer number not equal two. PGPRLayer expects two input layers in order, eg: acc layer & jerk layer')
        l_a = self.input_layers[0]
        #input_layers[1] may not be l_jerk. Below I use jp denotes jerk_processed
        l_j = weight_tiled_layer

        self.num_not_pred = num_not_pred
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.grad_clipping = grad_clipping
        self.mixing=mixing

        # (batch_size, seq_len, frame_dim)
        if self.input_shapes[0][0]!=self.input_shapes[1][0]:
            raise ValueError("Mismatch: all input shapes should have same batch size")
        self.batch_size, self.seq_len_a, self.frame_dim_a = self.input_shapes[0]
        self.batch_size, self.seq_len_jp, self.frame_dim_jp = self.input_shapes[1]

        # numbers of factors and maps should be inferred from weight tired layer: l_jerk
        # number of left factor == right factor 
        self.num_factors = l_j.num_factors # l_jerk.num_factors 
        # constructed acc should has same dim with acc 
        self.num_maps    = self.frame_dim_a

        # tire weights with l_jerk's weight
        W_x = l_j.W_x   
        W_y = l_j.W_m.T
        W_m = l_j.W_y.T
        # tile b_m with l_acc's b_m or not
        b_m = (l_a.b_m if b_m_tiled else init.Constant(0.))

        # init FactorGateLayer instance, None nonlinearity is applied in l_m
        self.l_a_i = InputLayer(shape=(self.batch_size, self.frame_dim_a), name='PGPRLayer l_a_i')
        self.l_jp_i = InputLayer(shape=(self.batch_size, self.frame_dim_jp), name='PGPRLayer l_jp_i')
        self.l_m   = FactorGateLayer([self.l_a_i, self.l_jp_i], num_factors=self.num_factors, num_maps=self.num_maps, W_x=W_x, W_y=W_y, W_m=W_m, b_m=b_m, nonlinearity=None, name='PGPRLayer l_m')

    def get_output_for(self, inputs, **kwargs):
        #inputs: l_a:(batch, time, dim) l_jp:(batch, time, dim)

        # Prepare inputs data 
        # (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
        inputs[0]=inputs[0].dimshuffle(1,0,2)
        inputs[1]=inputs[1].dimshuffle(1,0,2)

        # Slice acc_out[2-1,:,:]
        a = inputs[0][self.num_not_pred-1,:,:]
        # get all timesteps from jerk_processd 
        jp = inputs[1]

        # loop 
        def step(a_n, jp_n, *args):
             # computer maps units
            maps = helper.get_output(self.l_m, {self.l_a_i:a_n, self.l_jp_i:jp_n})

            # Clip gradients
            if self.grad_clipping is not False:
                maps = theano.gradient.grad_clip(
                    maps, -self.grad_clipping, self.grad_clipping)

            return maps
        # The l_m params are unchange during scan
        # non_seqs = helper.get_all_params(self.l_m) is not enough to get all params         
        # So we need specific l_m's params 
        non_seqs = [self.l_m.W_x, self.l_m.W_y, self.l_m.W_m, self.l_m.b_m]
        a_r = theano.scan(
            fn=step,
            sequences=[a,jp],
            non_sequences=non_seqs,
            strict=True)[0]

        # mixing acc and acc_r
        if self.mixing:
            a_r = (self.mixing)*a_r + (1-self.mixing)*inputs[0][self.num_not_pred:,:,:]  

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        a_r = a_r.dimshuffle(1, 0, 2)

        # applying nonliearity at here but not at step func
        return self.nonlinearity(a_r)

    def get_output_shape_for(self, input_shapes):
        # batch, timestep, framedim
        return input_shapes[0][0], input_shapes[0][1]-self.num_not_pred, self.num_maps

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(PGPRLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.l_m, **tags)
        return params

def build_PGP(seqs_shape, num_all_units, num_not_pred, noise):
    # Forward propagate  mapping units 
    l_i = InputLayer(shape=seqs_shape) 
    l_n = DropoutLayer(l_i, p=noise) 

    l_v = PGPLayer(l_n, num_factors=num_all_units['v_f'], num_maps=num_all_units['v_m'], nonlinearity=None, name='vec layer')
    l_v_n = NonlinearityLayer(l_v, nonlinearity=nonlinearities.sigmoid)

    l_a = PGPLayer(l_v_n, num_factors=num_all_units['a_f'], num_maps=num_all_units['a_m'], nonlinearity=None, name='acc layer')
    l_a_n = NonlinearityLayer(l_a, nonlinearity=nonlinearities.sigmoid)

    l_j = PGPLayer(l_a_n, num_factors=num_all_units['j_f'], num_maps=num_all_units['j_m'], nonlinearity=None, name='jerk layer')
    l_j_p = JerkLayer(l_j, nonlinearity=nonlinearities.sigmoid, name='post-jerk layer')
    
    # reconstruct a, v and vis
    l_a_r = PGPRLayer([l_a,l_j_p], num_not_pred=num_not_pred-2, b_m_tiled=True,  weight_tiled_layer=l_j, mixing=l_j_p.autonomy, name='acc reconstr layer') 
    l_v_r = PGPRLayer([l_v,l_a_r], num_not_pred=num_not_pred-1, b_m_tiled=True,  weight_tiled_layer=l_a, mixing=l_j_p.autonomy, name='vec reconstr layer')
    l_i_r = PGPRLayer([l_i,l_v_r], num_not_pred=num_not_pred, b_m_tiled=False, weight_tiled_layer=l_v, mixing=False, nonlinearity=None, name='vis reconstr layer')

    return dict(v=l_v, v_n=l_v_n,  a=l_a, j=l_j, i_r=l_i_r)

def build_PGP_2layers(seqs_shape, num_all_units, num_not_pred, noise):
    # Forward computer mapping units 
    l_i = InputLayer(shape=seqs_shape) 
    l_n = DropoutLayer(l_i, p=noise) 
    l_v = PGPLayer(l_n, num_factors=num_all_units['v_f'], num_maps=num_all_units['v_m'], name='vec layer')
    l_vp= JerkLayer(l_v, name='post vec layer')
    # reconstruct a, v and vis
    l_i_r = PGPRLayer([l_i,l_vp], num_not_pred=num_not_pred, b_m_tiled=False, weight_tiled_layer=l_v, mixing=False, name='vis reconstr layer')

    return dict(v=l_v, i_r=l_i_r)

def create_PGP_iter_functions( dataset, seqs_shape, num_all_units, layers, num_not_pred, momentum=0.9):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_size, seq_len, frame_dim  = seqs_shape

    # form an theano input var X and set test value for debug
    X_batch = T.tensor3('X_batch')
    X_batch.tag.test_value = np.random.randn(400, 16, 10).astype(np.float32)
    X = T.tensor3('X')
    X.tag.test_value = np.random.randn(50000, 16, 10).astype(np.float32)

    # get output for X
    X_batch_r  = helper.get_output(layers['i_r'], X_batch, deterministic=False)
    X_batch_r_t= helper.get_output(layers['i_r'], X_batch, deterministic=True)
    #i_r = helper.get_output(layers['i_r'], X, deterministic=True)
    v_o = helper.get_output(layers['v_n'], X, deterministic=True)


    # loss
    loss_train = squared_error(X_batch_r,   X_batch[:,num_not_pred:,:]).mean()
    loss_eval  = squared_error(X_batch_r_t, X_batch[:,num_not_pred:,:]).mean()

    # form batch slice
    batch_index = T.iscalar('batch_index')
    batch_index.tag.test_value = 0
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
    
    # form gradient update
    all_params = helper.get_all_params(layers['i_r'])
    learning_rate = T.scalar('learning_rate',dtype=theano.config.floatX)
    learning_rate.tag.test_value = 0.01
    updates = nesterov_momentum(loss_train, all_params, learning_rate, momentum)
    #updates = lasagne.updates.adam(loss_train, all_params, learning_rate)

    # dump funcs
    #iter_valid = theano.function_dump(
    #        'func_dump.pkl',
    #    [batch_index], [loss_eval],
    #    givens={
    #        X_batch: dataset['X_valid'][batch_slice],
    #    },
    #)

    # form 3 training functions
    iter_train = theano.function(
        [batch_index, learning_rate], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
        },
    )
    iter_valid = theano.function(
        [batch_index], [loss_eval],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval],
        givens={
            X_batch: dataset['X_test'][batch_slice],
        },
    )

    # i_r_fun = theano.function([X], i_r)
    v_func = theano.function([X], v_o,)

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        #vis_reconstruc=i_r_fun, 
        vec_out=v_func
    )

    ## prediction

    ## forward propagate X_batch 
    #v = helper.get_output(layers['vec'], X_batch, deterministic=False)
    #a = layers['acc'].get_output_for(v)
    #j = layers['jerk'].get_output_for(a)
    #j_rnn = layers['rnn'].get_output_for(j)
    ## (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
    #X_batch=X_batch.dimshuffle(1,0,2); v=v.dimshuffle(1,0,2); a=a.dimshuffle(1,0,2)
    #j=j.dimshuffle(1,0,2); j_rnn=j_rnn.dimshuffle(1,0,2)
    ## 2d list to hold gate computer results 
    #vis=[None]*seq_len;vec=[None]*seq_len;acc=[None]*seq_len;jerk=[None]*seq_len

    ## init gate layer, [TODO]don't foget indenpendent params 3*b
    ## j,a--->a
    #l_i_j = InputLayer(shape=(batch_size, num_all_units['j_m']),name='l_i_j')
    #l_i_a0= InputLayer(shape=(batch_size, num_all_units['a_m']),name='l_i_a0')
    #l_a_r = FactorGateLayer([l_i_j,l_i_a0], num_factors=num_all_units['j_f'],num_maps=num_all_units['a_m'], W_x=layers['jerk'].W_m.T, W_y=layers['jerk'].W_x, W_m=layers['jerk'].W_y.T) 
    ## a,v--->v
    #l_i_a1= InputLayer(shape=(batch_size, num_all_units['a_m']),name='l_i_a1')
    #l_i_v0= InputLayer(shape=(batch_size, num_all_units['v_m']),name='l_i_v0')
    #l_v_r = FactorGateLayer([l_i_a1,l_i_v0], num_factors=num_all_units['a_f'],num_maps=num_all_units['v_m'], W_x=layers['acc'].W_m.T, W_y=layers['acc'].W_x, W_m=layers['acc'].W_y.T) 
    ## v,vis--->vis
    #l_i_v1= InputLayer(shape=(batch_size, num_all_units['v_m']),name='l_i_v1')
    #l_i_vis= InputLayer(shape=(batch_size, num_all_units['vis']),name='l_i_vis')
    #l_vis_r = FactorGateLayer([l_i_v1,l_i_vis], num_factors=num_all_units['v_f'],num_maps=num_all_units['vis'], W_x=layers['vec'].W_m.T, W_y=layers['vec'].W_x, W_m=layers['vec'].W_y.T) 
   
    ## computer gate triangles
    #for t in range(4, seq_len):
    #    #TODO acc[t] = helper.get_output(l_a_r,{l_i_j:,l_i_a0:})
    #    pass

def main_PGP(dataset):
    batch_size = 400
    # (batchsize, seqlen, framedim)
    seqs_shape = (batch_size,dataset['timesteps'],dataset['frame_dim']) 
    num_all_units= dict(vis=16, v_f=200,v_m=100,a_f=100,a_m=50,j_f=10,j_m=10,j_rnn_h_units=10)
    num_not_pred = 4 # vision frames reserved for infer remaining frames.

    # build GAE model
    layers = build_PGP(seqs_shape, num_all_units, num_not_pred)
    # build iter funcs
    iter_funcs =  create_PGP_iter_functions(dataset, seqs_shape, num_all_units, layers, num_not_pred)
    
    print("Starting training...")
    now = time.time()
    num_epochs = 500
    try:
        for epoch in train(iter_funcs, dataset, learning_rate=0.01):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            #print("  validation accuracy:\t\t{:.2f} %".format( epoch['valid_accuracy'] * 100))
           # if epoch['number'] % 10 == 0:
           #     learning_rate -= 0.005

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass


###################
#GAE BEGIN
################## 
def build_GAE(frames_shapes, num_fac, num_maps, noise=0.3):
    frame_dim = frames_shapes[1]
    #l_x_i = InputLayer(frames_shapes, name='l_x_i')
    #l_x   = DropoutLayer(l_x_i, p=noise)
    #l_x   = DenseLayer(l_x, num_units=num_fac, W=init.GlorotNormal(), b=None,
    #                    nonlinearity=None, name='l_x_dense')

    #l_y_i = InputLayer(frames_shapes, name='l_y_i')
    #l_y   = DropoutLayer(l_y_i, p=noise)
    #l_y   = DenseLayer(l_y, num_units=num_fac, W=init.GlorotNormal(), b=None, 
    #                    nonlinearity=None, name='l_y_dense')

    #l_m   = GateTriangleLayer([l_x, l_y], num_units=num_maps, 
    #                    W=init.GlorotUniform(), b=init.Constant(0.),
    #                    nonlinearity=nonlinearities.sigmoid, name='l_m') 
    ##l_m_t = TransposedDenseLayer(l_m, W=l_m.W.T, num_units=num_fac, b=None, nonlinearity=None)
    #l_m_t = DenseLayer(l_m, num_units=num_fac, W=init.GlorotNormal(), b=None,
    #                    nonlinearity=None, name='l_m_t')
    #l_x_r = TransposedGateTriangleLayer([l_y, l_m_t], 
    #                    num_units=frame_dim, W=l_x.W.T, b=init.Constant(0.), 
    #                    nonlinearity=None, name='l_x_r',)
    #l_y_r = TransposedGateTriangleLayer([l_x, l_m_t], 
    #                    num_units=frame_dim, W=l_y.W.T, b=init.Constant(0.), 
    #                    nonlinearity=None, name='l_y_r',)
    #return dict(x_i=l_x_i, y_i=l_y_i, x=l_x, y=l_y, m=l_m, m_t=l_m_t, x_r=l_x_r, y_r=l_y_r)

    l_x_i = InputLayer(frames_shapes, name='l_x_i')
    l_x   = DropoutLayer(l_x_i, p=noise)
    l_y_i = InputLayer(frames_shapes, name='l_y_i')
    l_y   = DropoutLayer(l_y_i, p=noise)
    l_m   = FactorGateLayer([l_x,l_y], num_factors=num_fac, num_maps=num_maps,
                            W_x=init.GlorotUniform(), W_y=init.GlorotUniform(), 
                            W_m=init.GlorotUniform(), b_m=init.Constant(0.), 
                            nonlinearity=nonlinearities.sigmoid, name='l_m')

    l_x_r = FactorGateLayer([l_m, l_y], num_factors=num_fac, num_maps=frame_dim, 
                            W_x=init.GlorotUniform(), W_y=l_m.W_y, W_m=l_m.W_x.T, 
                            nonlinearity=None, name='l_x_r') 
    l_y_r = FactorGateLayer([l_m, l_x], num_factors=num_fac, num_maps=frame_dim, 
                            W_x=l_x_r.W_x, W_y=l_m.W_x, W_m=l_m.W_y.T, 
                            nonlinearity=None, name='l_y_r') 
    return dict(x_i=l_x_i, y_i=l_y_i, m=l_m, x_r=l_x_r, y_r=l_y_r)

def create_GAE_iter_functions(dataset, layers, batch_size, momentum, pretrain):
    # create 2 sybmol var for adjacent frames: x and y
    X_batch = T.matrix('X_batch')
    Y_batch = T.matrix('Y_batch')
    X_batch.tag.test_value = np.random.randn( 400, 10).astype(np.float32)
    Y_batch.tag.test_value = np.random.randn( 400, 10).astype(np.float32)
    # slice input data into batch
    batch_index = T.iscalar('batch_index')
    batch_index.tag.test_value = 0
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    # get network output
    l_x_r=layers['x_r']; l_y_r=layers['y_r']; l_x_i=layers['x_i']; l_y_i=layers['y_i'] 
    X_batch_r, Y_batch_r = helper.get_output([l_x_r, l_y_r], {l_x_i:X_batch,l_y_i:Y_batch}, deterministic=False)
    X_batch_r_t, Y_batch_r_t = helper.get_output([l_x_r, l_y_r], {l_x_i:X_batch,l_y_i:Y_batch}, deterministic=True)
    # computer loss
    loss_train = 0.5*squared_error(X_batch_r, X_batch).sum(axis=1).mean(axis=0) + 0.5*squared_error(Y_batch_r, Y_batch).sum(axis=1).mean(axis=0)
    loss_eval = 0.5*squared_error(X_batch_r_t, X_batch).sum(axis=1).mean(axis=0) + 0.5*squared_error(Y_batch_r_t, Y_batch).sum(axis=1).mean(axis=0)

    # update params with grident
    all_params = helper.get_all_params([l_x_r, l_y_r])
    learning_rate = T.scalar('learning_rate',dtype=theano.config.floatX)
    learning_rate.tag.test_value = 0.01
    updates = nesterov_momentum(loss_train, all_params, learning_rate, momentum)
    #updates = lasagne.updates.adam(loss_train, all_params, learning_rate)
    
    # build 3 functions
    iter_train = theano.function(
        [batch_index, learning_rate], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            Y_batch: dataset['Y_train'][batch_slice],
        },
    )

    if not pretrain:
        iter_valid = theano.function(
        [batch_index], [loss_eval],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            Y_batch: dataset['Y_valid'][batch_slice],
        },
        )

        iter_test = theano.function(
        [batch_index], [loss_eval],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            Y_batch: dataset['Y_test'][batch_slice],
        },
        )
    else:
        iter_valid=None
        iter_test=None

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    ) 

def main_GAE(dataset, num_epochs, batch_size, num_fac, num_maps, noise, learning_rate, pretrain):
    # (batchsize=100, timesteps=2, framedim=10), GAE dataset always have timesteps=2.
    frames_shapes = (batch_size, dataset['frame_dim']) 

    # build GAE model
    layers = build_GAE(frames_shapes, num_fac=num_fac, num_maps=num_maps, noise=0.3)
    # build iter funcs
    iter_funcs = create_GAE_iter_functions(dataset, layers, batch_size=batch_size, momentum=0.9, pretrain=pretrain) 
    
    print("Starting training GAE ...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset, learning_rate=learning_rate, batch_size=batch_size, pretrain=pretrain):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            if not pretrain:
                print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            #print("  validation accuracy:\t\t{:.2f} %".format( epoch['valid_accuracy'] * 100))
           # if epoch['number'] % 10 == 0:
           #     learning_rate -= 0.005

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass
    return layers

def main():
    # load data
    GAE_data, PGP_data = make_sinwave()

    # (batchsize, timsteps, framedim)
   # num_train=PGP_data['num_train']; 
    batch_size=400
    timesteps=PGP_data['timesteps']; framedim=PGP_data['frame_dim'] 
    seqs_shape = (batch_size, timesteps, framedim)
    num_all_units= dict(vis=16, v_f=200,v_m=100,a_f=100,a_m=50,j_f=10,j_m=10,j_rnn_h_units=10)
    # TODO 4 or 2 
    num_not_pred = 4 # vision frames reserved for infer remaining frames.

    # build PGP network and iter functions
    PGP_layers = build_PGP(seqs_shape, num_all_units, num_not_pred, noise=0.0)
    PGP_funcs = create_PGP_iter_functions(PGP_data, seqs_shape, num_all_units, 
                               PGP_layers, num_not_pred, momentum=0.9)

    PRETRAIN = True 
    if PRETRAIN:
        # pretrain using GAE model
        # train vec
        vec_layers = main_GAE(GAE_data, num_epochs=10, batch_size=100, 
                              num_fac=num_all_units['v_f'], num_maps=num_all_units['v_m'], 
                              learning_rate=0.01, noise=0.3, pretrain=False)

        # set PGP vec layer weight 
        PGP_layers['v'].W_x.set_value(vec_layers['m'].W_x.get_value()*utils.floatX(0.5))
        PGP_layers['v'].W_y.set_value(vec_layers['m'].W_y.get_value()*utils.floatX(0.5))
        PGP_layers['v'].W_m.set_value(vec_layers['x_r'].W_x.get_value().T)
        PGP_layers['v'].b_m.set_value(vec_layers['m'].b_m.get_value())

        # propagateing traing data through vec layer of PGP network to get vec_output
        v_o = PGP_funcs['vec_out'](PGP_data['X_train_np'])

        # prepare v_o data for acc pretrain
        frame_dim = v_o.shape[2] # 100
        v_o = v_o.reshape(v_o.shape[0], v_o.shape[1]*v_o.shape[2])
        seq_len=v_o.shape[1] # timesteps 15 * framedim 100 
        v_o = np.concatenate([v_o[i, 2*j*frame_dim:2*(j+1)*frame_dim][None,:] for j in range(seq_len/(frame_dim*2)) for i in range(PGP_data['num_train'])],0)
        v_o = v_o[np.random.permutation(v_o.shape[0])]
        X_v_o = v_o[:, :frame_dim]
        Y_v_o = v_o[:, frame_dim:]
        acc_data =  dict(
                        num_train=X_v_o.shape[0],
                        timesteps=2,
                        frame_dim=X_v_o.shape[1],
                        X_train=theano.shared(utils.floatX(X_v_o)),
                        Y_train=theano.shared(utils.floatX(Y_v_o)),
                        ) 
        # train acc 
        acc_layers = main_GAE(acc_data, num_epochs=10, batch_size=100, 
                              num_fac=num_all_units['a_f'], num_maps=num_all_units['a_m'], 
                             learning_rate=0.01, noise=0.3, pretrain=True)

        # set PGP acc layer weight 
        PGP_layers['a'].W_x.set_value(acc_layers['m'].W_x.get_value())
        PGP_layers['a'].W_y.set_value(acc_layers['m'].W_y.get_value())
        PGP_layers['a'].W_m.set_value(acc_layers['x_r'].W_x.get_value().T)
        PGP_layers['a'].b_m.set_value(acc_layers['m'].b_m.get_value())

    # train PGP model
    print("Starting training PGP ...")
    now = time.time()
    num_epochs = 500
    try:  # 2layer learing_rate=0.0005  
        for epoch in train(PGP_funcs, PGP_data, learning_rate=0.001,batch_size=batch_size, pretrain=False):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            #print("  validation accuracy:\t\t{:.2f} %".format( epoch['valid_accuracy'] * 100))
           # if epoch['number'] % 10 == 0:
           #     learning_rate -= 0.005

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    # plot
    #print 'Ploting ...'
    #plt.figure(1)
    #plt.axis('tight')
    #truth = plt.plot(PGP_data['X_train_np'].reshape(num_train, timesteps*framedim)[0],'b-', label='ground truth')
    #plt.ylabel('sin(x)')
    #vis
    #prediction = plt.plot(prediction.flatten(), 'g-', label='prediction')


if __name__ == '__main__':
    main()


##########
# Below codes are deprecated 
#########
class GateTriangleLayer(ElemwiseMergeLayer):
    def __init__(self, incomings, num_units, W=init.GlorotUniform(), b=init.Constant(0.), nonlinearity=nonlinearities.sigmoid, **kwargs):
        # incoming layers check
        if len(incomings) != 2:
            raise ValueError('Incoming layer number not equal two. GateTriangle expects two input layers in order')
        super(GateTriangleLayer, self).__init__(incomings, merge_function=T.mul, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        num_inputs = self.input_shapes[0][1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_for(self, inputs, **kwargs):
        # use super get_out_for to get elemwise multiplying
        _input = super(GateTriangleLayer, self).get_output_for(inputs, **kwargs)
        
        activation = T.dot(_input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        # (batch_size, num_units)
        return (input_shapes[0][0], self.num_units)

class TransposedGateTriangleLayer(ElemwiseMergeLayer):
    def __init__(self, incomings, num_units, W, b=init.Constant(0.), nonlinearity=nonlinearities.sigmoid, **kwargs):
        super(TransposedGateTriangleLayer, self).__init__(incomings, merge_function=T.mul, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)
        #num_inputs = self.input_shapes[0][1]
        self.num_units = num_units
        self.W = W
        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_for(self, inputs, **kwargs):
        _input = super(TransposedGateTriangleLayer, self).get_output_for(inputs, **kwargs)
        activation = T.dot(_input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return (input_shapes[0][0], self.num_units)

class PGPLayer_old_copyXY(Layer):
    """
    expect input have shape (batch_size, seq_len, frame_dim)
    """
    def __init__(self, incoming,
                     num_factors,
                     num_maps,
                     W_x=init.GlorotUniform(), 
                     W_y=init.GlorotUniform(),
                     W_m=init.GlorotUniform(),
                     b_m=init.Constant(0.),
                     nonlinearity=nonlinearities.sigmoid,
                     grad_clipping=False,
                     **kwargs):
        super(PGPLayer, self).__init__(incoming, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.grad_clipping = grad_clipping
        self.num_factors = num_factors
        self.num_maps = num_maps
        # (batch_size, seq_len, frame_dim)
        self.batch_size, self.seq_len, self.frame_dim = self.input_shape

        # init FactorGateLayer instance
        self.l_x_i = InputLayer(shape=(self.batch_size, self.frame_dim), name='l_x_i')
        self.l_y_i = InputLayer(shape=(self.batch_size, self.frame_dim), name='l_y_i')
        self.l_m   = FactorGateLayer([self.l_x_i, self.l_y_i], num_factors=self.num_factors, num_maps=self.num_maps, W_x=W_x, W_y=W_y, W_m=W_m, b_m=b_m, name='FactorGateLayer_l_m')

        # Make child layer parameters intuitively accessible
        self.W_x=self.l_m.W_x; self.W_y=self.l_m.W_y; self.W_m=self.l_m.W_m; self.b_m=self.l_m.b_m

    def get_output_for(self, input, **kwargs):
        # (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Prepare inputdata: stack adjenct frames horizontally
        _input=[]
        for i in range(self.input_shape[1]-1):
            _input.append(T.concatenate([input[i], input[i+1]], 1))
        input_xy = T.as_tensor_variable(_input)    
                
        # Create single recurrent computation step function
        def step(input_xy_n, *args):
            # slice x y from input_xy
            input_x_n = input_xy_n[:,:self.frame_dim]
            input_y_n = input_xy_n[:,self.frame_dim:]

            # computer maps units
            maps = helper.get_output(self.l_m, {self.l_x_i:input_x_n, self.l_y_i:input_y_n})

            # Clip gradients
            if self.grad_clipping is not False:
                maps = theano.gradient.grad_clip(
                    maps, -self.grad_clipping, self.grad_clipping)

            return maps

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.l_m)
      
        # Scan op iterates over first dimension of input and repeatedly applies the step function
        maps_out = theano.scan(
            fn=step,
            sequences=input_xy,
            non_sequences=non_seqs,
            strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        maps_out = maps_out.dimshuffle(1, 0, 2)

        return maps_out

    def get_output_shape_for(self, input_shape):
        # mapping units have a number less than input units number
        return input_shape[0], input_shape[1]-1, self.num_maps

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(PGPLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.l_m, **tags)
        return params

class JerkLayer_old(Layer):
    def __init__(self, incoming, nonlinearity,  **kwargs):
        super(JerkLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)

        # build dynamic between jerks' timesteps. create scalar (ndim=0) (shape=()) with init value 0.5
        self.autonomy = self.add_param(utils.floatX(0.5), shape=(), name='autonomy', regularizable=False)
        # transform autonomy to interval (0,1)
        self.autonomy = T.nnet.sigmoid(self.autonomy)

    def get_output_for(self, input, **kwargs):
        # (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        def step(j_n, prev_j, *args):
            return prev_j*self.autonomy + j_n*(1-self.autonomy)

        out = theano.scan(
            fn=step,
            sequences=input[1:,:,:], # discard first timestep
            outputs_info=input[0,:,:], # use first timestep
            non_sequences=[self.autonomy],
            strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        o = out.dimshuffle(1, 0, 2)
        
        return self.nonlinearity(o)

    def get_output_shape_for(self, input_shape):
        # jerk units have a number less than input  number
        return input_shape[0], input_shape[1]-1, input_shape[2] 

