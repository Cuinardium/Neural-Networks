from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def call(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        pass

    @abstractmethod
    def derivative(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        pass


# ----- Cross Entropy -----
class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def call(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        loss = -np.sum(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        ) / len(predicted)
        return loss

    def derivative(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return (predicted - actual) / (predicted * (1 - predicted)) / len(predicted)

# ----- Mean Squared Error -----
class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__()

    def call(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean((predicted - actual) ** 2)

    def derivative(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (predicted - actual) / len(predicted)
