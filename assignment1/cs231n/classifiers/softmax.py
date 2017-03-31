import numpy as np
from random import shuffle

def softmax_predict(W,X):
    scores = X.dot(W)
    predicted_class = np.argmax(scores,axis = 1)
    
    return predicted_class

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  num_train = X.shape[0]
  num_classes = W.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  for i in xrange(num_train):
      for j in range(num_classes):
        if j == y[i]:
            correct_score = exp_scores[i,j]
      
            
    
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores)
  exp_scores = np.exp(scores)
  probs = exp_scores/np.sum(exp_scores,axis = 1,keepdims = True)
  correct_log_probs = -np.log(probs[range(num_train),y])
  data_loss = np.sum(correct_log_probs)/num_train
  reg_loss = np.sum(W*W)
  loss = data_loss + reg_loss

  dscores = probs
  dscores[range(num_train),y] -=1
  dscores = dscores/num_train
  
  dW = X.T.dot(dscores)
  db = np.sum(dscores,axis=0,keepdims=True)
  dW += reg*W
  dW[0,:] = db
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW




