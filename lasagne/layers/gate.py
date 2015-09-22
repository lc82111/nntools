# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

from .. import nonlinearities 
from .. import utils
from .. import init
from .base import Layer 
from .merge import MergeLayer 
from .merge import ElemwiseMergeLayer 
from . import helper 
from .input import InputLayer

__all__ = [
    "PGPLayer",
    "RecurrentSoftmaxLayer",
    "AverageLayer",
]


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

        # Make child layer parameters intuitively accessible
        self.W_x=self.l_m.W_x; self.W_y=self.l_m.W_y; self.W_m=self.l_m.W_m; self.b_m=self.l_m.b_m

    def get_output_for(self, inputs, **kwargs):
        #inputs: l_a:(batch, time, dim) l_jp:(batch, time, dim)

        # Prepare inputs data 
        # (n_batch, n_time_steps, n_features)---->(n_time_steps, n_batch, n_features)
        inputs[0]=inputs[0].dimshuffle(1,0,2)
        inputs[1]=inputs[1].dimshuffle(1,0,2)

        # Slice acc_out[2-1:,:,:]
        a = inputs[0][self.num_not_pred-1:,:,:]
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


# Helper class
class RecurrentSoftmaxLayer(Layer):
    def __init__(self, incoming, num_units,
                 W=init.GlorotUniform(), 
                 b=init.Constant(0.),
		**kwargs):
        super(RecurrentSoftmaxLayer, self).__init__(incoming)
	self.num_time_steps = self.input_shape[1]
	self.num_features = self.input_shape[2]	
	self.num_units = num_units
	self.W = self.add_param(W, (self.num_features, self.num_units), name="W")
	self.b = self.add_param(b, (self.num_units, ), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
	return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        batchsize, _, _ = input.shape
	activation = T.tensordot(input, self.W, axes=1) + self.b
	activation = activation.reshape((self.input_shape[0] * self.input_shape[1], self.num_units), ndim=2)
	#activation = activation.reshape(( -1, self.num_units), ndim=2)
	result = T.nnet.softmax(activation)
	result = result.reshape((self.input_shape[0], self.input_shape[1], self.num_units))
	#result = result.reshape((batchsize, self.input_shape[1], self.num_units), ndim=3)
	return T.mean(result, axis=1)

class AverageLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(AverageLayer, self).__init__(incoming)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_output_for(self, input, *args, **kwargs):
        return input.mean(axis=1)
