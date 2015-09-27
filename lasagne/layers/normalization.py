# -*- coding: utf-8 -*-
"""
Preliminary implementation of batch normalization for Lasagne.
Does not include a way to properly compute the normalization factors over the
full training set for testing, but can be used as a drop-in for training and
validation.

Author: Jan Schlüter
"""

from .base import Layer 
from .. import nonlinearities 
from .. import init
import theano
import theano.tensor as T

__all__ = [
    "BatchNormLayer",
    "batch_norm",
    "LocalResponseNormalization2DLayer"
]

class BatchNormLayer(Layer):
    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
            nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).
        
        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = self.add_param(init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std

        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        #normalized = theano.tensor.nnet.bn.batch_normalization(input, gamma, beta, mean, std) 
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)

def batch_norm(layer):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).

    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity)


class LocalResponseNormalization2DLayer(Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    Input order is assumed to be `BC01`.

    If the value of the ith channel is :math:`x_i`, the output is

    .. math::

        x_i = \frac{x_i}{ (k + ( \alpha \sum_j x_j^2 ))^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    This code is adapted from pylearn2.
    """

    def __init__(self, incoming, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        """
        :parameters:
            - incoming: input layer or shape
            - alpha: see equation above
            - k: see equation above
            - beta: see equation above
            - n: number of adjacent channels to normalize over.
        """
        super(LocalResponseNormalization2DLayer, self).__init__(incoming,
                                                                **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return input / scale
