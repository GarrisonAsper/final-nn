# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        
        Z_curr = np.dot(W_curr, A_prev) + b_curr #linear transformation

        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError('Unsuported activation function')
        
        #for debugging
        if activation == 'sigmoid':
            print(f"Applying Sigmoid in layer with shape {Z_curr.shape}")

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from _single_forward for use in backprop.
        """
        cache = {'A0': X.T} #stores input activations

        A_prev = X.T
        for idx, layer in enumerate(self.arch):
            W_curr = self._param_dict[f'W{idx+1}']
            b_curr = self._param_dict[f'b{idx+1}']
            activation = layer['activation']

            #actually do the forward pass
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            #store results
            cache[f'Z{idx+1}'] = Z_curr
            cache[f'A{idx+1}'] = A_curr
            
            #update A for next layer
            A_prev = A_curr

        #print statemement for debugging
        #print(f"Shape of forward output y_hat: {A_curr.shape}")
        
        return A_curr, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        #compute dZ for both activation functions
        if activation_curr =='relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr =='sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise ValueError('Unsupported activation function')
        
        #batch size
        m = A_prev.shape[1]

        #print statement for debugging
        #print(f"Shape of W_curr in _single_backprop: {W_curr.shape}")

        #calculating gradients
        dA_prev = np.dot(W_curr.T, dZ_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis = 1, keepdims = True) / m

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        #dictionary to store gradients
        grad_dict = {}
    
        #compute the gradient from loss function
        if self._loss_func == 'binary_cross_entropy':
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError('Unsupported loss function')
        
        #iterate backwards through layers
        for layer_idx in reversed(range(1, len(self.arch) + 1)):
            A_prev = cache[f'A{layer_idx -1}']
            Z_curr = cache[f'Z{layer_idx}']

            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']
            activation_curr = self.arch[layer_idx -1]['activation']

            #print statement for debugging
            #print(f"dA_curr shape before _single_backprop: {dA_prev.shape}")
            #compute gradients
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_prev, activation_curr)

            #store gradients
            grad_dict[f'dW{layer_idx}'] = dW_curr
            grad_dict[f'db{layer_idx}'] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer_idx in range(1, len(self.arch) + 1):
            #get gradients
            dW_curr = grad_dict[f'dW{layer_idx}']
            db_curr = grad_dict[f'db{layer_idx}']

            print(f"Layer {layer_idx}: Max Weight Gradient: {np.max(np.abs(dW_curr))}")
            print(f"Layer {layer_idx}: Max Bias Gradient: {np.max(np.abs(db_curr))}")

            #descend gradient
            self._param_dict[f'W{layer_idx}'] -= self._lr * dW_curr
            self._param_dict[f'b{layer_idx}'] -= self._lr * db_curr

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        for epoch in range(self._epochs):
            #training
            y_hat_train, cache_train = self.forward(X_train)

            #compute loss
            if self._loss_func == 'binary_cross_entropy':
                loss_train = self._binary_cross_entropy(y_train, y_hat_train)
            elif self._loss_func == 'mean_squared_error':
                loss_train = self._mean_squared_error(y_train, y_hat_train)
            else:
                raise ValueError('Unsupported loss function')
            
            #backpropagation
            grad_dict = self.backprop(y_train, y_hat_train, cache_train)

            #update parameters
            self._update_params(grad_dict)

            #store loss
            per_epoch_loss_train.append(loss_train)

            #forward pass for validation
            y_hat_val, _ = self.forward(X_val)

            #computing validation loss
            if self._loss_func == 'binary_cross_entropy':
                loss_val = self._binary_cross_entropy(y_val, y_hat_val)
            elif self._loss_func == 'mean_squared_error':
                loss_val = self._mean_squared_error(y_val, y_hat_val)
            
            #store validation loss
            per_epoch_loss_val.append(loss_val)

            #print loss every 10 epochs just to check in and make sure its all Gucci
            if epoch % 10 == 0 or epoch == self._epochs -1:
                print(f'Epoch {epoch+1} / {self._epochs}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}')

        return per_epoch_loss_train, per_epoch_loss_val
            

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        #compute prediction with forward pass
        y_hat, _ = self.forward(X)

        #apply threshold for classification tasks
        if self._loss_func == 'binary_cross_entropy':
            y_hat = (y_hat >= 0.5).astype(int) #converts to 0 or 1

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        output = 1 / (1 + np.exp(-Z))

        print(f"Sigmoid Output: Min: {np.min(output)}, Max: {np.max(output)}")

        return output

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #computing activation
        sigmoid_Z = self._sigmoid(Z)

        #compute derivative
        dSigmoid = sigmoid_Z * (1 - sigmoid_Z)
    
        dZ = dA * dSigmoid

        #print statements for debugging   
        #print(f"Shape of sigmoid_Z: {sigmoid_Z.shape}")
        #print(f"Shape of dA in _sigmoid_backprop: {dA.shape}")
        #print(f"Shape of dSigmoid: {dSigmoid.shape}")
        #print(f"Shape of dZ before return: {dZ.shape}")

        return np.array(dZ, copy = True)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0,Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        #calculating derivative
        dZ = np.array(dA, copy = True) #using copy to keep proper shape
        dZ[Z<=0] = 0

        return dZ


    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #staying away from log(0)
        epsilon = 1e-8
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        # Ensure y and y_hat have the same shape
        if y.shape != y_hat.shape:
            y = y.reshape(y_hat.shape)

        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        epsilon = 1e-8  # Prevent log(0)
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        if y.shape != y_hat.shape:
            y = y.reshape(y_hat.shape)

        # Compute correct gradient normalization
        dA = (y_hat - y) / y.shape[0]  #normalized by number of samples in batch

        # Debugging print
        print(f"BCE Gradients: Min {np.min(dA)}, Max {np.max(dA)}")

        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.mean((y - y_hat.T) ** 2)

        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = y.shape[0]

        #compute derivative
        dA = (2 / m) * (y_hat -y.T)
        #print statement for debugging
        #print(f"Shape of dA in loss backprop: {dA.shape}")

        return dA