import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class test_nn2(object):
  def __init__(self, input_dim=(3, 32, 32),
               num_conv=2, num_filters=[32,32], filter_size=[3,3], conv_stride=[1,1],
               pool_size=[2,2], pool_stride=[2,2],
               num_affine=1, hidden_dim=[100], num_classes=10, weight_scale=1e-5, reg=0.0,
               dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    
    self.params['W1'] =np.random.randn(np.prod(input_dim),num_classes)*weight_scale
    self.params['b1'] =np.zeros([1,num_classes])
    
  def loss(self, X, y=None):
    cache = {}
    scores = None
    
    scores, cache['scores'] = affine_forward(X, self.params['W1'],self.params['b1'])

    if y is None:
      return scores
    
    loss, grads = 0, {}    
    
    loss, dscores = softmax_loss(scores,y)    
    dx, grads['W1'], grads['b1'] = affine_backward(dscores,cache['scores'])
    

    grads['W1'] += self.reg*self.params['W1']
    loss += 0.5*self.reg*np.sum(self.params['W1']* self.params['W1'])
    return loss, grads


class test_nn(object):
  """
  CNN using form of:
  [conv-relu-pool]XN - [affine]XM - [softmax]
  """

  def __init__(self, input_dim=(3, 32, 32),
               num_conv=2, num_filters=[32,32], filter_size=[3,3], conv_stride=[1,1],
               pool_size=[2,2], pool_stride=[2,2],
               num_affine=1, hidden_dim=[100], num_classes=10, weight_scale=1e-5, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_conv: Number of convolution-relu-pool layers    
    - num_filters: Number of filters to use in the convolutional layer, tuple of length num_conv
    - filter_size: Size of filters to use in the convolutional layer, are always square, tuple of length num_conv
    - conv_stride: stride for convolution, tuple of length num_conv
    
    - pool_size: Tuple of the size of the pooling operation at each layer, always square, tuple of length num_conv
    - pool_stride: Stride for the pooling operation at each layer, tuple of length num_conv
    
    - num_affine: Number of fully connected affine layers at the end
    - hidden_dim: Number of units to use in the fully-connected hidden layer, tuple of length num_affine
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    
    It is assumed:
    - pool layers will always be 2x2 with stride 2
    - convolution stride = 1
    """
    self.params = {}
    self.conv_params = {}
    self.pool_params = {}
    
    self.reg = reg
    self.dtype = dtype
    
    self.num_conv = num_conv
    self.num_affine = num_affine

    for k, v in self.params.iteritems():
        self.params[k] = v.astype(dtype)
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    for layer in range(num_conv): # initialize all the convolution weights
        layer_num = layer+1 # counter for the dicts that hold the params, practice is to index from 1
        if layer == 0:
            up_c, up_h, up_w = input_dim
        
#        pad = (filter_size[layer] - 1) / 2
#        self.conv_params['conv_param'+str(layer_num)] = {'stride': conv_stride[layer], 'pad':pad}
#        self.pool_params['pool_param'+str(layer_num)] = {'pool_height': pool_size[layer], 'pool_width': pool_size[layer], 'stride': pool_stride[layer]}

#        self.params['W'+str(layer_num)] = np.random.randn(num_filters[layer],up_c,filter_size[layer],filter_size[layer])*weight_scale
#        self.params['b'+str(layer_num)] = np.zeros(num_filters[layer])
       
        # Determine what comes out of the convolution:
#        up_c = num_filters[layer]
#        up_h = 1 + (up_h + 2* pad - filter_size[layer])/conv_stride[layer]
#        up_w = 1 + (up_w + 2* pad - filter_size[layer])/conv_stride[layer]
        
        # Determine what comes out of the pool:
#        up_h = 1 + (up_h - pool_size[layer]) / pool_stride[layer]
#        up_w = 1 + (up_w - pool_size[layer]) / pool_stride[layer]
        
        
    # find the number of connections that we need for the first affine layer  
    # account for the debug case where num_conv = 0
    if num_conv == 0:
        num_up = np.prod(input_dim)
#    else: 
#        num_up = up_c*up_h*up_w
    
    
#    for layer in range(num_affine):
#        layer_num = num_conv+layer+1
#        self.params['W'+str(layer_num)] =np.random.randn(num_up,hidden_dim[layer])*weight_scale
#        self.params['b'+str(layer_num)] =np.zeros([1,hidden_dim[layer]])
#        num_up = hidden_dim[layer] # index forward for the next affine layer
#        
    # account for debug case where num_affine = 0
    if num_affine == 0:
        num_up = num_up
        layer_num = 0

    # Make the last affine layer before the softmax
    self.params['W'+str(layer_num+1)] =np.random.randn(num_up,num_classes)*weight_scale
    self.params['b'+str(layer_num+1)] =np.zeros([1,num_classes])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    cache = {}
    
#    for layer in range(self.num_conv):
#        layer_num = layer + 1
#        X, cache['cache'+str(layer_num)] = conv_relu_pool_forward(X, self.params['W'+str(layer_num)],
#                                                               self.params['b'+str(layer_num)],
#                                                               self.conv_params['conv_param'+str(layer_num)],
#                                                               self.pool_params['pool_param'+str(layer_num)])
    # debug case where num_conv = 0
    if self.num_conv == 0:
        layer = 0

#    for layer in range(self.num_affine):
#        layer_num = layer + self.num_conv + 1
#        X, cache['cache'+str(layer_num)] = affine_relu_forward(X, self.params['W'+str(layer_num)],
#                                                               self.params['b'+str(layer_num)])
    # debug case where num_affine = 0
    if self.num_affine == 0:
        layer_num = self.num_conv + self.num_affine
        
    scores, cache['scores'] = affine_forward(X, self.params['W'+str(layer_num+1)],self.params['b'+str(layer_num+1)])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    loss, dscores = softmax_loss(scores,y)
    num_layers = self.num_conv + self.num_affine + 1
    
    dx, grads['W'+str(num_layers)], grads['b'+str(num_layers)] = affine_backward(dscores,cache['scores'])
    #print "Backwards pass:"
    #print "layer: " +'W'+str(num_layers)
    
#    for layer in range(num_layers-1, num_layers-1- self.num_affine,-1):
#        dx, grads['W'+str(layer)], grads['b'+str(layer)] = affine_relu_backward(dx,cache['cache'+str(layer)])
#        #print "layer: " + 'W'+str(layer)

#    for layer in range(layer-1,0,-1):
#        dx, grads['W'+str(layer)], grads['b'+str(layer)] = conv_relu_pool_backward(dx,cache['cache'+str(layer)])
        #print "layer: " + 'W'+str(layer)
        
    #find out the regularization loss
    for layer, weight in self.params.iteritems():
        if str(layer).startswith('W'):
            loss += 0.5*self.reg*np.sum(weight*weight) # add up the loss
            grads[str(layer)] += self.reg*weight # regularize the gradient

   
    #loss, dscores = softmax_loss(scores,y)
    #dx3, grads['W3'], grads['b3'] = affine_backward(dscores,cache_scores)
    #dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, cache_affine)
    #dx1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2, cache_conv)
    #
    #loss += 0.5* self.reg*(np.sum(self.params['W1']*self.params['W1']) + 
    #                      np.sum(self.params['W2']*self.params['W2']) + 
    #                      np.sum(self.params['W3']*self.params['W3']))
    #grads['W1'] += self.reg*self.params['W1']
    #grads['W2'] += self.reg*self.params['W2']
    #grads['W3'] += self.reg*self.params['W3']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
