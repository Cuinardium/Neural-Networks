from numpy import ndarray
import numpy as np

class SM():

    def softmax(self, input: ndarray):
        return np.exp(input) / np.sum(np.exp(input))

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        # Para guardar los valores de entrada y salida
        self.input = np.zeros(input_size)
        self.output = np.zeros(output_size)

        self.weights = np.random.randn(input_size, output_size)


    def foward_prop(self, input: ndarray):
        self.input = input

        excitements = input @ self.weights
        activations = self.softmax(excitements)

        self.output = activations

        return activations


    def back_prop(self, loss_gradient:ndarray, learning_rate: float):

        # Loss en funcion de los pesos
        gradient_weights = self.input.T @ loss_gradient

        # Por ahora gradient descent
        self.weights -= learning_rate * gradient_weights

        # Loss en funcion de la entrada, 
        # con softmax no hay que multiplicar por la derivada (se simplifica)
        gradient_input = loss_gradient @ self.weights.T

        return gradient_input
