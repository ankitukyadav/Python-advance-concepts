"""
Advanced Python: Machine Learning Framework Implementation
This script demonstrates building a machine learning framework from scratch,
showcasing advanced numerical computing, optimization, and design patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable, Dict, Any, Union
from abc import ABC, abstractmethod
import pickle
import json
from dataclasses import dataclass
from enum import Enum
import time

# Type aliases
Matrix = np.ndarray
Vector = np.ndarray

# Activation functions
class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, x: Matrix) -> Matrix:
        """Backward pass (derivative)."""
        pass

class ReLU(ActivationFunction):
    """Rectified Linear Unit activation function."""
    
    def forward(self, x: Matrix) -> Matrix:
        return np.maximum(0, x)
    
    def backward(self, x: Matrix) -> Matrix:
        return (x > 0).astype(float)

class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""
    
    def forward(self, x: Matrix) -> Matrix:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def backward(self, x: Matrix) -> Matrix:
        s = self.forward(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function."""
    
    def forward(self, x: Matrix) -> Matrix:
        return np.tanh(x)
    
    def backward(self, x: Matrix) -> Matrix:
        return 1 - np.tanh(x) ** 2

class Softmax(ActivationFunction):
    """Softmax activation function."""
    
    def forward(self, x: Matrix) -> Matrix:
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x: Matrix) -> Matrix:
        s = self.forward(x)
        return s * (1 - s)

# Loss functions
class LossFunction(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_pred: Matrix, y_true: Matrix) -> float:
        """Calculate loss."""
        pass
    
    @abstractmethod
    def backward(self, y_pred: Matrix, y_true: Matrix) -> Matrix:
        """Calculate gradient."""
        pass

class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function."""
    
    def forward(self, y_pred: Matrix, y_true: Matrix) -> float:
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: Matrix, y_true: Matrix) -> Matrix:
        return 2 * (y_pred - y_true) / y_pred.shape[0]

class CrossEntropy(LossFunction):
    """Cross-entropy loss function."""
    
    def forward(self, y_pred: Matrix, y_true: Matrix) -> float:
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped))
    
    def backward(self, y_pred: Matrix, y_true: Matrix) -> Matrix:
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred_clipped) / y_pred.shape[0]

# Optimizers
class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    @abstractmethod
    def update(self, params: Dict[str, Matrix], gradients: Dict[str, Matrix]) -> None:
        """Update parameters using gradients."""
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params: Dict[str, Matrix], gradients: Dict[str, Matrix]) -> None:
        for name, param in params.items():
            if name not in self.velocities:
                self.velocities[name] = np.zeros_like(param)
            
            self.velocities[name] = (self.momentum * self.velocities[name] - 
                                   self.learning_rate * gradients[name])
            params[name] += self.velocities[name]

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params: Dict[str, Matrix], gradients: Dict[str, Matrix]) -> None:
        self.t += 1
        
        for name, param in params.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # Update biased first and second moment estimates
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradients[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (gradients[name] ** 2)
            
            # Bias correction
            m_corrected = self.m[name] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[name] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

# Neural network layers
class Layer(ABC):
    """Abstract base class for neural network layers."""
    
    @abstractmethod
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Matrix]:
        """Get layer parameters."""
        pass
    
    @abstractmethod
    def get_gradients(self) -> Dict[str, Matrix]:
        """Get parameter gradients."""
        pass

class Dense(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int, activation: Optional[ActivationFunction] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
        # Cache for backward pass
        self.last_input = None
        self.last_output = None
        
        # Gradients
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, x: Matrix) -> Matrix:
        self.last_input = x
        output = np.dot(x, self.weights) + self.bias
        
        if self.activation:
            output = self.activation.forward(output)
        
        self.last_output = output
        return output
    
    def backward(self, grad_output: Matrix) -> Matrix:
        if self.activation:
            grad_output = grad_output * self.activation.backward(self.last_output)
        
        # Calculate gradients
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        # Calculate gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def get_params(self) -> Dict[str, Matrix]:
        return {
            f'weights_{id(self)}': self.weights,
            f'bias_{id(self)}': self.bias
        }
    
    def get_gradients(self) -> Dict[str, Matrix]:
        return {
            f'weights_{id(self)}': self.grad_weights,
            f'bias_{id(self)}': self.grad_bias
        }

class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, dropout_rate: float = 0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x: Matrix) -> Matrix:
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad_output: Matrix) -> Matrix:
        if self.training and self.mask is not None:
            return grad_output * self.mask
        return grad_output
    
    def get_params(self) -> Dict[str, Matrix]:
        return {}
    
    def get_gradients(self) -> Dict[str, Matrix]:
        return {}
    
    def set_training(self, training: bool):
        self.training = training

# Neural Network
class NeuralNetwork:
    """Multi-layer neural network."""
    
    def __init__(self, layers: List[Layer], loss_function: LossFunction, optimizer: Optimizer):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.training_history = {'loss': [], 'accuracy': []}
    
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass through all layers."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output: Matrix) -> None:
        """Backward pass through all layers."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train_step(self, x: Matrix, y: Matrix) -> Tuple[float, float]:
        """Single training step."""
        # Set layers to training mode
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(True)
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Calculate loss
        loss = self.loss_function.forward(y_pred, y)
        
        # Calculate accuracy (for classification)
        accuracy = self._calculate_accuracy(y_pred, y)
        
        # Backward pass
        grad_loss = self.loss_function.backward(y_pred, y)
        self.backward(grad_loss)
        
        # Update parameters
        all_params = {}
        all_gradients = {}
        
        for layer in self.layers:
            params = layer.get_params()
            gradients = layer.get_gradients()
            all_params.update(params)
            all_gradients.update(gradients)
        
        self.optimizer.update(all_params, all_gradients)
        
        return loss, accuracy
    
    def predict(self, x: Matrix) -> Matrix:
        """Make predictions."""
        # Set layers to evaluation mode
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(False)
        
        return self.forward(x)
    
    def fit(self, x_train: Matrix, y_train: Matrix, epochs: int, batch_size: int = 32,
            x_val: Optional[Matrix] = None, y_val: Optional[Matrix] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Train the neural network."""
        n_samples = x_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss, accuracy = self.train_step(x_batch, y_batch)
                epoch_loss += loss
                epoch_accuracy += accuracy
            
            # Average metrics
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            
            # Validation
            val_loss, val_accuracy = None, None
            if x_val is not None and y_val is not None:
                val_pred = self.predict(x_val)
                val_loss = self.loss_function.forward(val_pred, y_val)
                val_accuracy = self._calculate_accuracy(val_pred, y_val)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
                if val_loss is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return self.training_history
    
    def _calculate_accuracy(self, y_pred: Matrix, y_true: Matrix) -> float:
        """Calculate accuracy for classification tasks."""
        if y_pred.shape[1] > 1:  # Multi-class
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y_true, axis=1)
            return np.mean(pred_classes == true_classes)
        else:  # Binary classification
            pred_classes = (y_pred > 0.5).astype(int)
            return np.mean(pred_classes == y_true)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'layers': [],
            'loss_function': type(self.loss_function).__name__,
            'optimizer': type(self.optimizer).__name__,
            'training_history': self.training_history
        }
        
        for layer in self.layers:
            layer_data = {
                'type': type(layer).__name__,
                'params': layer.get_params()
            }
            if hasattr(layer, 'input_size'):
                layer_data['input_size'] = layer.input_size
                layer_data['output_size'] = layer.output_size
            if hasattr(layer, 'dropout_rate'):
                layer_data['dropout_rate'] = layer.dropout_rate
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

# Data preprocessing utilities
class DataPreprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize(x: Matrix, method: str = 'minmax') -> Tuple[Matrix, Dict[str, Any]]:
        """Normalize data."""
        if method == 'minmax':
            x_min = np.min(x, axis=0)
            x_max = np.max(x, axis=0)
            x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
            params = {'min': x_min, 'max': x_max}
        elif method == 'zscore':
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
            x_normalized = (x - x_mean) / (x_std + 1e-8)
            params = {'mean': x_mean, 'std': x_std}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return x_normalized, params
    
    @staticmethod
    def one_hot_encode(y: Vector, num_classes: Optional[int] = None) -> Matrix:
        """One-hot encode labels."""
        if num_classes is None:
            num_classes = int(np.max(y)) + 1
        
        encoded = np.zeros((len(y), num_classes))
        encoded[np.arange(len(y)), y.astype(int)] = 1
        return encoded
    
    @staticmethod
    def train_test_split(x: Matrix, y: Matrix, test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
        """Split data into training and testing sets."""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = x.shape[0]
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

# Example datasets
class DatasetGenerator:
    """Generate example datasets for testing."""
    
    @staticmethod
    def make_classification(n_samples: int = 1000, n_features: int = 2, n_classes: int = 2,
                          random_state: Optional[int] = None) -> Tuple[Matrix, Vector]:
        """Generate a random classification dataset."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate random centers for each class
        centers = np.random.randn(n_classes, n_features) * 2
        
        x = []
        y = []
                
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            # Generate samples around each center
            class_samples = np.random.randn(samples_per_class, n_features) * 0.8 + centers[class_idx]
            x.append(class_samples)
            y.extend([class_idx] * samples_per_class)
        
        x = np.vstack(x)
        y = np.array(y)
        
        # Shuffle the data
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]
    
    @staticmethod
    def make_regression(n_samples: int = 1000, n_features: int = 1, noise: float = 0.1,
                       random_state: Optional[int] = None) -> Tuple[Matrix, Matrix]:
        """Generate a random regression dataset."""
        if random_state is not None:
            np.random.seed(random_state)
        
        x = np.random.randn(n_samples, n_features)
        
        # Create a non-linear relationship
        y = np.sum(x**2, axis=1) + np.sin(np.sum(x, axis=1)) + noise * np.random.randn(n_samples)
        y = y.reshape(-1, 1)
        
        return x, y
    
    @staticmethod
    def make_spiral(n_samples: int = 1000, n_classes: int = 2, 
                   random_state: Optional[int] = None) -> Tuple[Matrix, Vector]:
        """Generate a spiral dataset."""
        if random_state is not None:
            np.random.seed(random_state)
        
        samples_per_class = n_samples // n_classes
        x = []
        y = []
        
        for class_idx in range(n_classes):
            r = np.linspace(0.1, 1, samples_per_class)
            theta = np.linspace(class_idx * 2 * np.pi / n_classes, 
                              (class_idx + 1) * 2 * np.pi / n_classes + 2 * np.pi, 
                              samples_per_class)
            
            x_class = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            x_class += 0.1 * np.random.randn(samples_per_class, 2)
            
            x.append(x_class)
            y.extend([class_idx] * samples_per_class)
        
        x = np.vstack(x)
        y = np.array(y)
        
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

# Model evaluation metrics
class Metrics:
    """Model evaluation metrics."""
    
    @staticmethod
    def accuracy(y_true: Matrix, y_pred: Matrix) -> float:
        """Calculate accuracy."""
        if y_pred.shape[1] > 1:  # Multi-class
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y_true, axis=1)
        else:  # Binary
            pred_classes = (y_pred > 0.5).astype(int).flatten()
            true_classes = y_true.flatten()
        
        return np.mean(pred_classes == true_classes)
    
    @staticmethod
    def precision_recall_f1(y_true: Matrix, y_pred: Matrix) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for binary classification."""
        if y_pred.shape[1] > 1:
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y_true, axis=1)
        else:
            pred_classes = (y_pred > 0.5).astype(int).flatten()
            true_classes = y_true.flatten()
        
        tp = np.sum((pred_classes == 1) & (true_classes == 1))
        fp = np.sum((pred_classes == 1) & (true_classes == 0))
        fn = np.sum((pred_classes == 0) & (true_classes == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return precision, recall, f1
    
    @staticmethod
    def mean_squared_error(y_true: Matrix, y_pred: Matrix) -> float:
        """Calculate mean squared error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def r2_score(y_true: Matrix, y_pred: Matrix) -> float:
        """Calculate R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

# Visualization utilities
class Visualizer:
    """Visualization utilities for ML models."""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    @staticmethod
    def plot_decision_boundary(model: NeuralNetwork, x: Matrix, y: Vector, 
                             resolution: int = 100, save_path: Optional[str] = None):
        """Plot decision boundary for 2D classification."""
        if x.shape[1] != 2:
            raise ValueError("Decision boundary plotting only supports 2D data")
        
        # Create a mesh
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(mesh_points)
        
        if predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (predictions > 0.5).astype(int).flatten()
        
        predictions = predictions.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Model builder utility
class ModelBuilder:
    """Utility class for building common model architectures."""
    
    @staticmethod
    def build_mlp(input_size: int, hidden_sizes: List[int], output_size: int,
                  activation: str = 'relu', output_activation: Optional[str] = None,
                  dropout_rate: float = 0.0) -> List[Layer]:
        """Build a multi-layer perceptron."""
        activation_map = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'softmax': Softmax()
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(Dense(prev_size, hidden_size, activation_map[activation]))
            
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        output_act = activation_map.get(output_activation) if output_activation else None
        layers.append(Dense(prev_size, output_size, output_act))
        
        return layers
    
    @staticmethod
    def build_classifier(input_size: int, num_classes: int, hidden_sizes: List[int] = None,
                        dropout_rate: float = 0.0) -> NeuralNetwork:
        """Build a classification model."""
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
        output_size = num_classes if num_classes > 2 else 1
        
        layers = ModelBuilder.build_mlp(
            input_size, hidden_sizes, output_size,
            activation='relu', output_activation=output_activation,
            dropout_rate=dropout_rate
        )
        
        loss_function = CrossEntropy() if num_classes > 2 else MeanSquaredError()
        optimizer = Adam()
        
        return NeuralNetwork(layers, loss_function, optimizer)
    
    @staticmethod
    def build_regressor(input_size: int, hidden_sizes: List[int] = None,
                       dropout_rate: float = 0.0) -> NeuralNetwork:
        """Build a regression model."""
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        layers = ModelBuilder.build_mlp(
            input_size, hidden_sizes, 1,
            activation='relu', dropout_rate=dropout_rate
        )
        
        loss_function = MeanSquaredError()
        optimizer = Adam()
        
        return NeuralNetwork(layers, loss_function, optimizer)

# Cross-validation utility
class CrossValidator:
    """K-fold cross-validation utility."""
    
    def __init__(self, k: int = 5, random_state: Optional[int] = None):
        self.k = k
        self.random_state = random_state
    
    def split(self, x: Matrix, y: Matrix) -> List[Tuple[Matrix, Matrix, Matrix, Matrix]]:
        """Generate k-fold splits."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = x.shape[0]
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // self.k
        
        splits = []
        
        for i in range(self.k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.k - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            splits.append((x_train, x_test, y_train, y_test))
        
        return splits
    
    def evaluate_model(self, model_builder_func: Callable, x: Matrix, y: Matrix,
                      epochs: int = 100, verbose: bool = False) -> Dict[str, float]:
        """Evaluate model using cross-validation."""
        scores = []
        
        for fold, (x_train, x_test, y_train, y_test) in enumerate(self.split(x, y)):
            if verbose:
                print(f"Training fold {fold + 1}/{self.k}")
            
            # Build and train model
            model = model_builder_func()
            model.fit(x_train, y_train, epochs=epochs, verbose=False)
            
            # Evaluate
            y_pred = model.predict(x_test)
            accuracy = Metrics.accuracy(y_test, y_pred)
            scores.append(accuracy)
            
            if verbose:
                print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }

# Demonstration functions
def demonstrate_classification():
    """Demonstrate classification with the ML framework."""
    print("=== Classification Demo ===")
    
    # Generate dataset
    x, y = DatasetGenerator.make_classification(n_samples=1000, n_features=2, 
                                              n_classes=3, random_state=42)
    
    # Preprocess data
    x_normalized, _ = DataPreprocessor.normalize(x, method='zscore')
    y_encoded = DataPreprocessor.one_hot_encode(y)
    
    # Split data
    x_train, x_test, y_train, y_test = DataPreprocessor.train_test_split(
        x_normalized, y_encoded, test_size=0.2, random_state=42
    )
    
    # Build model
    model = ModelBuilder.build_classifier(
        input_size=2, num_classes=3, hidden_sizes=[64, 32], dropout_rate=0.1
    )
    
    print(f"Training on {x_train.shape[0]} samples, testing on {x_test.shape[0]} samples")
    
    # Train model
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, 
                       x_val=x_test, y_val=y_test, verbose=True)
    
    # Evaluate
    y_pred = model.predict(x_test)
    accuracy = Metrics.accuracy(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Visualize results
    print("Plotting training history...")
    Visualizer.plot_training_history(history)
    
    return model, x_test, y_test

def demonstrate_regression():
    """Demonstrate regression with the ML framework."""
    print("\n=== Regression Demo ===")
    
    # Generate dataset
    x, y = DatasetGenerator.make_regression(n_samples=1000, n_features=1, 
                                          noise=0.1, random_state=42)
    
    # Preprocess data
    x_normalized, _ = DataPreprocessor.normalize(x, method='zscore')
    y_normalized, _ = DataPreprocessor.normalize(y, method='zscore')
    
    # Split data
    x_train, x_test, y_train, y_test = DataPreprocessor.train_test_split(
        x_normalized, y_normalized, test_size=0.2, random_state=42
    )
    
    # Build model
    model = ModelBuilder.build_regressor(
        input_size=1, hidden_sizes=[64, 32], dropout_rate=0.1
    )
    
    print(f"Training on {x_train.shape[0]} samples, testing on {x_test.shape[0]} samples")
    
    # Train model
    history = model.fit(x_train, y_train, epochs=150, batch_size=32, 
                       x_val=x_test, y_val=y_test, verbose=True)
    
    # Evaluate
    y_pred = model.predict(x_test)
    mse = Metrics.mean_squared_error(y_test, y_pred)
    r2 = Metrics.r2_score(y_test, y_pred)
    
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_cross_validation():
    """Demonstrate cross-validation."""
    print("\n=== Cross-Validation Demo ===")
    
    # Generate dataset
    x, y = DatasetGenerator.make_spiral(n_samples=500, n_classes=2, random_state=42)
    
    # Preprocess data
    x_normalized, _ = DataPreprocessor.normalize(x, method='zscore')
    y_encoded = DataPreprocessor.one_hot_encode(y)
    
    # Define model builder function
    def build_model():
        return ModelBuilder.build_classifier(
            input_size=2, num_classes=2, hidden_sizes=[32, 16], dropout_rate=0.1
        )
    
    # Perform cross-validation
    cv = CrossValidator(k=5, random_state=42)
    results = cv.evaluate_model(build_model, x_normalized, y_encoded, 
                               epochs=50, verbose=True)
    
    print(f"\nCross-validation results:")
    print(f"Mean accuracy: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in results['scores']]}")

def demonstrate_optimizer_comparison():
    """Demonstrate different optimizers."""
    print("\n=== Optimizer Comparison Demo ===")
    
    # Generate dataset
    x, y = DatasetGenerator.make_classification(n_samples=800, n_features=2, 
                                              n_classes=2, random_state=42)
    
    # Preprocess data
    x_normalized, _ = DataPreprocessor.normalize(x, method='zscore')
    y_encoded = y.reshape(-1, 1).astype(float)  # Binary classification
    
    # Split data
    x_train, x_test, y_train, y_test = DataPreprocessor.train_test_split(
        x_normalized, y_encoded, test_size=0.2, random_state=42
    )
    
    # Test different optimizers
    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'SGD + Momentum': SGD(learning_rate=0.01, momentum=0.9),
        'Adam': Adam(learning_rate=0.001)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}...")
        
        # Build model
        layers = ModelBuilder.build_mlp(2, [32, 16], 1, activation='relu', 
                                       output_activation='sigmoid')
        model = NeuralNetwork(layers, MeanSquaredError(), optimizer)
        
        # Train
        history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=False)
        
        # Evaluate
        y_pred = model.predict(x_test)
        accuracy = Metrics.accuracy(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'history': history
        }
        
        print(f"{name} - Test Accuracy: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history']['loss'], label=name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['history']['accuracy'], label=name)
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demonstration function."""
    print("=== Advanced Python: Machine Learning Framework Demo ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_classification()
    demonstrate_regression()
    demonstrate_cross_validation()
    demonstrate_optimizer_comparison()
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
