import numpy as np

from Model.layers import *
from Model.layer_utils import *

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    for i in range(self.num_layers):
        if i == 0:
            self.params['W%d'%(i+1)] = np.random.randn(input_dim, hidden_dims[i]) * weight_scale
            self.params['b%d'%(i+1)] = np.zeros(hidden_dims[i])
            if self.use_batchnorm:
                self.params['gamma%d'%(i+1)] = np.ones(hidden_dims[i])
                self.params['beta%d'%(i+1)] = np.zeros(hidden_dims[i])
        elif i == (self.num_layers - 1):
            self.params['W%d'%(i+1)] = np.random.randn(hidden_dims[i-1], num_classes) * weight_scale
            self.params['b%d'%(i+1)] = np.zeros(num_classes)
        else:
            self.params['W%d'%(i+1)] = np.random.randn(hidden_dims[i-1],hidden_dims[i]) * weight_scale
            self.params['b%d'%(i+1)] = np.zeros(hidden_dims[i])
            if self.use_batchnorm:
                self.params['gamma%d'%(i+1)] = np.ones(hidden_dims[i])
                self.params['beta%d'%(i+1)] = np.zeros(hidden_dims[i])

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    hidden_layer = {}
    hidden_cache = {}
    dropout_cache = {}
    num_layers = self.num_layers
    for i in range(num_layers-1):
        if self.use_batchnorm:  
            if i == 0:
                hidden_layer['%d'%(i+1)], hidden_cache['%d'%(i+1)] = affine_norm_relu_forward(X, self.params['W%d'%(i+1)], self.params['b%d'%(i+1)], self.params['gamma%d'%(i+1)], self.params['beta%d'%(i+1)], self.bn_params[i])
            else:
                hidden_layer['%d'%(i+1)], hidden_cache['%d'%(i+1)] = affine_norm_relu_forward(hidden_layer['%d'%i], self.params['W%d'%(i+1)], self.params['b%d'%(i+1)], self.params['gamma%d'%(i+1)], self.params['beta%d'%(i+1)], self.bn_params[i])
        else:
            if i == 0:
                hidden_layer['%d'%(i+1)], hidden_cache['%d'%(i+1)] = affine_relu_forward(X, self.params['W%d'%(i+1)], self.params['b%d'%(i+1)])
            else:
                hidden_layer['%d'%(i+1)], hidden_cache['%d'%(i+1)] = affine_relu_forward(hidden_layer['%d'%i], self.params['W%d'%(i+1)], self.params['b%d'%(i+1)])
        if self.use_dropout:
            hidden_layer['%d'%(i+1)], dropout_cache['%d'%(i+1)] = dropout_forward(hidden_layer['%d'%(i+1)], self.dropout_param)
    scores, scores_cache = affine_forward(hidden_layer['%d'%(num_layers-1)], self.params['W%d'%num_layers], self.params['b%d'%num_layers])

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads, dhidden = 0.0, {}, {}
    loss, dscores = softmax_loss(scores, y)
    dhidden['%d'%(num_layers-1)], grads['W%d'%num_layers], grads['b%d'%num_layers] = affine_backward(dscores, scores_cache)
    for i in range(num_layers-1, 0, -1):
        if self.use_dropout:
            dhidden['%d'%i] = dropout_backward(dhidden['%d'%i], dropout_cache['%d'%i])
        if self.use_batchnorm:
            dhidden['%d'%(i-1)], grads['W%d'%i], grads['b%d'%i], grads['gamma%d'%i], grads['beta%d'%i] = affine_norm_relu_backward(dhidden['%d'%i], hidden_cache['%d'%i])
        else:     
            dhidden['%d'%(i-1)], grads['W%d'%i], grads['b%d'%i] = affine_relu_backward(dhidden['%d'%i], hidden_cache['%d'%i])
    
    for i in range(1, num_layers+1):
        loss += 0.5 * self.reg * np.sum(self.params['W%d'%i]**2)
        grads['W%d'%i] += self.reg * self.params['W%d'%i]

    return loss, grads
