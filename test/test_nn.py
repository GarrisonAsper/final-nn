import numpy as np
import pytest
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

# Sample architecture for testing
nn_arch = [
    {"input_dim": 4, "output_dim": 3, "activation": "relu"},
    {"input_dim": 3, "output_dim": 2, "activation": "relu"},
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
]

# Initialize a small neural network for unit testing
nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=2, epochs=1, loss_function="binary_cross_entropy")

# Sample input data
X_test = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])  # Shape (2, 4)
y_test = np.array([[1], [0]])  # Shape (2, 1)

def test_single_forward(activation):

    W = np.random.randn(3,4) * 0.1
    b = np.random.randn(3,1) * 0.1
    A_prev = X_test.T
    
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, activation)

    assert A_curr.shape == (3,2)
    assert Z_curr.shape == (3,2)

def test_forward():

    y_hat, cache = nn.forward(X_test)

    assert y_hat.shape == (1,2)
    assert "A0" in cache

def test_single_backprop():

    W = np.random.randn(3,4) * 0.1
    b = np.random.randn(3,1) * 0.1
    A_prev = X_test.T
    dA_curr = no.random.randn(3,2)

    A_curr, Z_curr = nn._single_forward(W, b, A_prev, 'relu')
    dA_prev, dW_curr, db_curr = nn._single_backprop(W, b , Z_curr, A_prev, dA_curr, 'relu')

    assert dA_prev.shape == (4,2)
    assert dW_curr.shape == (3,4)
    assert db_curr.shape == (3,1)

def test_predict():
    
    y_pred = nn.predict(X_test)

    assert y_pred,shape ==(1,2)
    assert np.all((y_pred ==0) | (y_pred ==1))

def test_binary_cross_entropy():
  
    y_hat = np.array([[0.9], [0.1]])
    loss = nn._binary_cross_entropy(y_test, y_hat)

    assert isinstance(loss) > 0

def test_binary_cross_entropy_backprop():
    
    y_hat = np.array([[0.9], [0.1]])
    dA = nn._binary_cross_entropy_backprop(y_test, y_hat)

    assert dA.shape == y_hat.shape
    assert np.all(dA < 1)

def test_mean_squared_error():
    y_hat = np.array([[0.8], [0.2]])
    loss = nn._mean_squared_error(y_test, y_hat)

    assert isinstance(loss, float)
    assert loss > 0 

def test_mean_squared_error_backprop():
    y_hat = np.array([[0.8], [0.2]])
    dA = nn._mean_squared_error_backprop(y_test, y_hat)

    assert dA.shape == y_hat.shape

def test_sample_seqs():
    sequences = ["AAA", "TTT", "CCC", "GGG"]
    labels = [1, 1, 0, 0]

    sampled_seqs, sampled_labels = sample_seqs(sequences, labels)

    assert len(sampled_seqs) == len(sampled_labels)
    assert sampled_labels.count(0) == sampled_labels.count(1)

def test_one_hot_encode_seqs():
    sequences = ["AT", "GC"]
    encoded = one_hot_encode_seqs(sequences)

    assert encoded.shape == (2, 8)
    assert np.all((encoded == 0) | (encoded == 1))