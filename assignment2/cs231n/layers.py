import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  
  num_train = x.shape[0]
  x_f = np.reshape(x,(num_train,-1))
  out = np.dot(x_f,w)+b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)

  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  num_train = x.shape[0]
  x_f = np.reshape(x,(num_train,-1))
    
  dw = np.dot(x_f.T,dout)
  db = np.sum(dout,axis=0, keepdims=True)
  dx = np.dot(dout,w.T)
  dx = np.reshape(dx,x.shape)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x<=0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, {}
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    #print "in train"
    mu_batch = np.mean(x,axis = 0)
    var_batch = np.sum((x-mu_batch)**2, axis = 0)/N
    
    x_centered = x-mu_batch
    sqrt_var = np.sqrt(var_batch + eps)
    
    x_norm = x_centered/sqrt_var
    out = gamma * x_norm + beta
    
    running_mean = momentum * running_mean + (1-momentum) * mu_batch
    running_var = momentum * running_var + (1-momentum) * var_batch
    
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    
    cache = (x,x_norm,gamma,beta,x_centered,sqrt_var,bn_param)
    #cache['x'] = x
    #cache['x_norm'] = x_normal
    #cache['gamma'] = gamma
    #cache['x_centered'] = x_centered
    #cache['sqrt_var'] = sqrt_var  
    #cache['bn_param'] = bn_param
       
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    x_normal = (x - running_mean) / (np.sqrt(running_var)+eps)
    out = gamma * x_normal + beta
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param

  

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  #N, D = cache['x'].shape
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x,x_norm,gamma,beta,x_centered,sqrt_var,bn_param = cache
  N,D = x.shape
  dx_norm = dout * gamma
  dvar = np.sum(dx_norm * x_centered * -0.5*sqrt_var**-3.0,axis=0)
  dmu = np.sum(dx_norm * -1/sqrt_var,axis = 0) + dvar * -2*np.sum(x_centered,axis = 0)/N
    
  dx = dx_norm * 1/sqrt_var + dvar * 2*x_centered/N + dmu/N
  dgamma = np.sum(dout * x_norm,axis = 0)
  dbeta = np.sum(dout,axis = 0)
  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None
  cache = {}

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p)/p
    out = x*mask
    cache['dropout_param'] = dropout_param
    cache['mask'] = mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout* cache[1]
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  N,C,H,W = x.shape  
  F,C,HH,WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
    
  # find how many w and h strides to make
  strides_col = (W-WW+2*pad)/stride+1
  strides_row = (H-HH+2*pad)/stride+1


  # initialize the out variable
  out = np.zeros([N,F,strides_row,strides_col])

  # create the padded input
  xp = np.zeros([N,C,H+2*pad,W+2*pad])
  xp[:,:,pad:H+pad,pad:W+pad] = x

  for n in xrange(0,N): #over every image
    for f in xrange(0,F): # over all the filters
        w_flat = w[f].flatten()
        for sr in range(0,strides_row):
            row_start = sr*stride
            row_end = sr*stride + HH
            
            for sc in range(0,strides_col):
                col_start = sc*stride
                col_end = sc*stride + WW
                #print "Row: "+str(row_start)+" - "+str(row_end)+"\tCol: "+str(col_start)+" - "+str(col_end)
                #print xp[0,row_start:row_end,col_start:col_end]
                
                out[n,f,sr,sc] = np.dot(xp[n,:,row_start:row_end,col_start:col_end].flatten(),w_flat) + b[f]
                               

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  """
  #w: Filter weights of shape (F, C, HH, WW)
    
  dx, dw, db = None, None, None
  x, w, b, conv_param =  cache
  N,C,H,W = x.shape  
  F,_,HH,WW = w.shape
  _,_,Hh,Hw = dout.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
    
  
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  db = np.sum(dout, axis = (0,2,3))
  dw = np.zeros(w.shape)

  # create the padded input
  xp = np.zeros([N,C,H+2*pad,W+2*pad])
  xp[:,:,pad:H+pad,pad:W+pad] = x

  # dw 
  for f in range(F):                  # loop over filters
      for channel in range(C):        # loop over channels in filter
        for i in range(HH):           # over the rows in w
            for j in range(WW):       # over the columns in w
                x_sub = xp[:,channel,i:i+Hh*stride:stride,j:j+Hw*stride:stride]
                dw[f,channel,i,j] = np.sum(x_sub*dout[:,f,:,:])
                #print "i: " + str(i) + "\tj: " + str(j) + "\tHp: " + str(Hp) + "\tWp: " + str(Wp) + "\tx: " + str(x.shape) + "\tstride: " + str(stride) 

  dx = np.zeros((N, C, H, W))
  for nprime in range(N):
    for i in range(H):
        for j in range(W):
            for f in range(F):
                for k in range(Hh):
                    for l in range(Hw):
                        mask1 = np.zeros_like(w[f, :, :, :])
                        mask2 = np.zeros_like(w[f, :, :, :])
                        if (i + pad - k * stride) < HH and (i + pad - k * stride) >= 0:
                            mask1[:, i + pad - k * stride, :] = 1.0
                        if (j + pad - l * stride) < WW and (j + pad - l * stride) >= 0:
                            mask2[:, :, j + pad - l * stride] = 1.0
                        w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                        dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked
   

    
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N,C,H,W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
    
  H1 = 1 + (H - pool_height) / stride
  W1 = 1 + (W - pool_width) / stride
    
  out = np.zeros((N,C,H1,W1))
  for n in range(N):
    for c in range(C):
        for h in range(H1):
            for w in range(W1):
                out[n,c,h,w] = np.max(x[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param = cache
    
  N,C,H,W = x.shape
  _,_,H1,W1 = dout.shape

  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  dx = np.zeros((N,C,H,W))
  for n in range(N):
    for c in range(C):
        for h in range(H1):
            for w in range(W1):
                idx = np.argmax(x[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width])
                rowNum = idx/pool_width
                colNum = idx%pool_width
                dx[n,c,h*stride+rowNum,w*stride+colNum] = dout[n,c,h,w]
                
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  bn_param['eps'] = eps
  momentum = bn_param.get('momentum', 0.9)
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  out, cache = None, {}

  if mode == 'train':
    mu = np.mean(x,axis = (0,2,3)).reshape(1,C,1,1)
    var = (np.sum((x-mu)**2, axis = (0,2,3))/(N*H*W)).reshape(1,C,1,1)
    
    xhat = (x-mu)/(np.sqrt(var+eps))
    out = gamma.reshape(1,C,1,1)*xhat + beta.reshape(1,C,1,1)
    
    running_mean = momentum * running_mean + (1-momentum) * np.squeeze(mu)
    running_var = momentum * running_var + (1-momentum) * np.squeeze(var)
    
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    cache = (bn_param, x, xhat, gamma, beta, mu, var)
  elif mode == 'test':
    mu = running_mean.reshape(1,C,1,1)
    var = running_var.reshape(1,C,1,1)
    xhat = (x - mu) / np.sqrt(var+eps)
    out = gamma.reshape(1,C,1,1) * xhat + beta.reshape(1,C,1,1)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  bn_param, x, xhat, gamma, beta, mu, var  = cache
  N,C,H,W = x.shape
  eps = bn_param['eps']

  dx_norm = dout * gamma.reshape(1,C,1,1)
  dvar = np.sum(dx_norm * (x-mu) * -0.5*np.sqrt(var + eps)**-3.0,axis=(0,2,3))
  dmu = np.sum(dx_norm * -1/np.sqrt(var + eps),axis = (0,2,3)) + dvar * -2*np.sum((x-mu),axis = (0,2,3))/(N*H*W)

  Nt = N*H*W
  dx = (1. / Nt) * gamma.reshape(1,C,1,1) * (var + eps)**(-1. / 2.) *(Nt * dout- np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1) - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1))
    
  dgamma = np.sum(dout * xhat,axis = (0,2,3))
  dbeta = np.sum(dout,axis = (0,2,3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
