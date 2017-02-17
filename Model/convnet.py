import numpy as np

from Model.layers import *
from Model.fast_layers import *
from Model.layer_utils import *


class ConvNet(object):
  """
  A multiple-layer convolutional network with the following architecture:
  
  (conv - [spatialbatchnorm] - relu - [max pool]) x N - (affine - [batchnorm] - relu - [dropout]) x M - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=[3, 32, 32], num_filters=[16, 16], filter_size=7,
               hidden_dim=[20, 20], num_classes=10, filter_stride=1, pad=None, 
               pool_size=2, pool_stride=2, weight_scale=1e-3, reg=0.0, 
               dtype=np.float32, seed=None, use_batchnorm=False, dropout=0.0):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: List [C, H, W] giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - filter_stride: Stride for convolution
    - pad: Number of zeros to be padded during convolution
    - pool_size: Size of pooling filters used in convolution
    - pool_stride: Stride for pooling in convolution
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    - use_batchnorm: Whether or not the network should use batch normalization
      including the convolution layers and full connected layers
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    """
    
    # deal with input parameters
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.dropout = dropout
    self.filter_stride = filter_stride
    self.pool_stride = pool_stride
    (C, H, W) = input_dim
    
    if type(filter_size) is list:
      (Hc, Wc) = filter_size
    else:
      Hc = filter_size
      Wc = filter_size
        
    if type(pool_size) is list:
      (Hp, Wp) = pool_size
    else:
      Hp = pool_size
      Wp = pool_size
    self.pool_height = Hp
    self.pool_width = Wp
        
    if not pad:
        pad = (filter_size - 1) / 2
    self.pad = pad
    
    if seed is not None:
      np.random.seed(seed)
    
    num_filters_arr = np.array(num_filters)
    num_filters_nonzero = num_filters_arr[np.nonzero(num_filters_arr)]
    num_conv_layer = num_filters_nonzero.shape[0]
    num_hidden_layer = len(hidden_dim)
    self.num_conv_layer = num_conv_layer
    self.num_hidden_layer = num_hidden_layer
    num_layer = num_conv_layer + num_hidden_layer + 1
    self.num_layer = num_layer
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # calculate each convolution layer output size
    H_conv = []
    W_conv = []
    pool_switch = []
    i = 0
    while i < len(num_filters):
      if i == 0:
        H_conv.append(1 + (H + 2 * pad - Hc) / filter_stride)
        W_conv.append(1 + (W + 2 * pad - Wc) / filter_stride)
      else:
        H_conv.append(1 + (H_conv[-1] + 2 * pad - Hc) / filter_stride)
        W_conv.append(1 + (W_conv[-1] + 2 * pad - Wc) / filter_stride)
      if i != len(num_filters)-1:
        if num_filters[i+1] == 0:
          H_conv[-1] = 1 + (H_conv[-1] - Hp) / pool_stride
          W_conv[-1] = 1 + (W_conv[-1] - Wp) / pool_stride
          i += 1
          pool_switch.append(1)
        else:
          pool_switch.append(0)
      else:
        pool_switch.append(0)
      i += 1
    self.pool_switch = pool_switch
    conv_out_dim = num_filters_nonzero[-1] * H_conv[-1] *  W_conv[-1]
    
    # initialize the weight and bias
    for i,j in enumerate(num_filters_nonzero):
      if i == 0:
        self.params['W'+str(i+1)] = np.random.randn(j, C, Hc, Wc) * weight_scale
        self.params['b'+str(i+1)] = np.zeros(j)
      else:
        self.params['W'+str(i+1)] = np.random.randn(j, num_filters_nonzero[i-1], Hc, Wc) * weight_scale
        self.params['b'+str(i+1)] = np.zeros(j)
        
    for i,j in enumerate(hidden_dim):
      if i == 0:
        self.params['W'+str(i+num_conv_layer+1)] = np.random.randn(conv_out_dim, j) * weight_scale
        self.params['b'+str(i+num_conv_layer+1)] = np.zeros(j)
      else:
        self.params['W'+str(i+num_conv_layer+1)] = np.random.randn(hidden_dim[i-1], j) * weight_scale
        self.params['b'+str(i+num_conv_layer+1)] = np.zeros(j)
        
    self.params['W'+str(num_layer)] = np.random.randn(hidden_dim[-1], num_classes) * weight_scale
    self.params['b'+str(num_layer)] = np.zeros(num_classes)
    
    # initialize the batch normalization scale and shift parameters
    if self.use_batchnorm:
      for i,j in enumerate(np.concatenate((num_filters_nonzero, hidden_dim), axis=0)):
        self.params['gamma'+str(i+1)] = np.ones(j)
        self.params['beta'+str(i+1)] = np.zeros(j)
    
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(num_layer-1)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the multiple-layer convolutional network.
    
    """
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = {'stride': self.filter_stride, 'pad': self.pad}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': self.pool_height, 'pool_width': self.pool_width, 'stride': self.pool_stride}
    
    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    mode = 'test' if y is None else 'train'
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    ## forward pass
    # convolutional layers
    conv = {}
    conv_cache = {}
    conv['L0'] = X
    num_conv_layer = self.num_conv_layer
    use_batchnorm = self.use_batchnorm
    pool_switch = self.pool_switch
    for i in range(1, num_conv_layer+1):
      if use_batchnorm and pool_switch[i-1] == 1:
        conv['L'+str(i)], conv_cache['L'+str(i)] = conv_norm_relu_pool_forward(conv['L'+str(i-1)], self.params['W'+str(i)], self.params['b'+str(i)], conv_param, pool_param, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
      elif not use_batchnorm and pool_switch[i-1] == 1:
        conv['L'+str(i)], conv_cache['L'+str(i)] = conv_relu_pool_forward(conv['L'+str(i-1)], self.params['W'+str(i)], self.params['b'+str(i)], conv_param, pool_param)
      elif use_batchnorm and pool_switch[i-1] == 0:
        conv['L'+str(i)], conv_cache['L'+str(i)] = conv_norm_relu_forward(conv['L'+str(i-1)], self.params['W'+str(i)], self.params['b'+str(i)], conv_param, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
      else:
        conv['L'+str(i)], conv_cache['L'+str(i)] = conv_relu_forward(conv['L'+str(i-1)], self.params['W'+str(i)], self.params['b'+str(i)], conv_param)
    
    # full connected layers
    hidden = {}
    hidden_cache = {}
    dropout_cache = {}
    hidden['L0'] = conv['L'+str(num_conv_layer)]
    for i in range(1, self.num_hidden_layer+1):
      if use_batchnorm:
        hidden['L'+str(i)], hidden_cache['L'+str(i)] = affine_norm_relu_forward(hidden['L'+str(i-1)], self.params['W'+str(i+num_conv_layer)], self.params['b'+str(i+num_conv_layer)], self.params['gamma'+str(i+num_conv_layer)], self.params['beta'+str(i+num_conv_layer)], self.bn_params[i+num_conv_layer-1])
      else:
        hidden['L'+str(i)], hidden_cache['L'+str(i)] = affine_relu_forward(hidden['L'+str(i-1)], self.params['W'+str(i+num_conv_layer)], self.params['b'+str(i+num_conv_layer)])
      if self.use_dropout:
        hidden['L'+str(i)], dropout_cache['L'+str(i)] = dropout_forward(hidden['L'+str(i)], self.dropout_param)
    
    # output layer
    num_layer = self.num_layer
    num_hidden_layer = self.num_hidden_layer
    scores, scores_cache = affine_forward(hidden['L'+str(num_hidden_layer)], self.params['W'+str(num_layer)], self.params['b'+str(num_layer)])
    
    if y is None:
      return scores
    
    # calculate output loss
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores,y)
    
    ## backward pass
    # output layer
    dhidden = {}
    dhidden['L'+str(num_hidden_layer)], grads['W'+str(num_layer)], grads['b'+str(num_layer)] = affine_backward(dscores, scores_cache)
    
    # full connected layers
    for i in range(num_hidden_layer, 0, -1):
      if self.use_dropout:
        dhidden['L'+str(i)] = dropout_backward(dhidden['L'+str(i)], dropout_cache['L'+str(i)])
      if self.use_batchnorm:
        dhidden['L'+str(i-1)], grads['W'+str(i+num_conv_layer)], grads['b'+str(i+num_conv_layer)], grads['gamma'+str(i+num_conv_layer)], grads['beta'+str(i+num_conv_layer)] = affine_norm_relu_backward(dhidden['L'+str(i)], hidden_cache['L'+str(i)])
      else:
        dhidden['L'+str(i-1)], grads['W'+str(i+num_conv_layer)], grads['b'+str(i+num_conv_layer)] = affine_relu_backward(dhidden['L'+str(i)], hidden_cache['L'+str(i)])
    
    # convolutional layers
    dconv = {}
    dconv['L'+str(num_conv_layer)] = dhidden['L0']
    for i in range(num_conv_layer, 0, -1):
      if use_batchnorm and pool_switch[i-1] == 1:
        dconv['L'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = conv_norm_relu_pool_backward(dconv['L'+str(i)], conv_cache['L'+str(i)])
      elif not use_batchnorm and pool_switch[i-1] == 1:
        dconv['L'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)] = conv_relu_pool_backward(dconv['L'+str(i)], conv_cache['L'+str(i)])
      elif use_batchnorm and pool_switch[i-1] == 0:
        dconv['L'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = conv_norm_relu_backward(dconv['L'+str(i)], conv_cache['L'+str(i)])
      else:
        dconv['L'+str(i-1)], grads['W'+str(i)], grads['b'+str(i)] = conv_relu_backward(dconv['L'+str(i)], conv_cache['L'+str(i)])
    
    # add the L2 regularization term
    for i in range(1, num_layer+1):
      loss += 0.5 * self.reg * np.sum(self.params['W%d'%i]**2)
      grads['W%d'%i] += self.reg * self.params['W%d'%i]
    
    return loss, grads

