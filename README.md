# Neural Networks in Python

A modular neural network library in Python supporting Convolutional Neural Networks (CNN), Multi-Layer Perceptrons (MLP), and Autoencoders. The project is designed for extensibility and experimentation, with a clean architecture for layers, activations, optimizers, and loss functions.

## Features

- **Convolutional Neural Network (CNN):** Modular implementation with support for convolutional, pooling, flatten, fully connected, and activation layers.
- **Multi-Layer Perceptron (MLP):** Fully connected feedforward network for classification and regression tasks.
- **Autoencoder:** Encoder-decoder architecture for unsupervised learning and dimensionality reduction.
- **Variational Autoencoder (VAE):** Probabilistic encoder-decoder model for generative learning, latent space exploration, and unsupervised representation learning. Includes KL-divergence regularization and supports β-VAE variants.
- **Customizable activation functions:** Sigmoid, Tanh, ReLU, and more.
- **Multiple optimization methods:** Gradient Descent, Momentum, Adaptive Eta, Adam.
- **Configurable via `config.json`.**
- **Visualization tools:** Confusion matrix, feature maps, filter visualization, error plots.
- **Dataset loader and result saving utilities.**

## Setup with Virtual Environment

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install the project in editable mode:**
   ```bash
   pip install -e .
   ```

4. **Run the main program:**
   ```bash
   python -m main
   ```

5. **Run all tests:**
   ```bash
   python -m unittest discover tests
   ```

6. **Run an example script:**
   ```bash
    python -m examples.compare_cnn_architectures
    python -m examples.vae_emoji
   ```


## Configuration

Edit `config.json` to set training parameters, activation functions, and optimizer settings. Example:
```json
{
  "epochs": 10,
  "fully_connected_activation": {
    "type": "sigmoid",
    "beta": 1.0
  },
  "optimizer": {
    "type": "adam",
    "eta": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08
  }
}
```
- `epochs`: Number of training epochs.
- `fully_connected_activation.type`: `sigmoid`, `tanh`, or `relu`.
- Optimizer types: `gradient_descent`, `momentum`, `adaptive_eta`, `adam`.

## Datasets and Results

- Download the shapes dataset and place it in a `data/` folder at the project root.
- All results (plots, predictions, logs) are saved in a `results/` folder at the project root.

## Project Structure

- `layers/`: Layer implementations (convolutional, pooling, fully connected, etc.)
- `networks/`: Network architectures (cnn, mlp, autoencoder)
- `utils/`: Utilities for data loading, plotting, saving
- `examples/`: Example scripts and experiments
- `tests/`: Unit tests

## References

- [CNNs, Part 1: An Introduction to Convolution Neural Networks](https://victorzhou.com/blog/intro-to-cnns-part-1/)
- [CNNs, Part 2: Training a Convolutional Neural Network](https://victorzhou.com/blog/intro-to-cnns-part-2/)
- [cnn-from-scratch repo](https://github.com/vzhou842/cnn-from-scratch/tree/forward-only)
- [Fast convolution](https://medium.com/@thepyprogrammer/2d-image-convolution-with-numpy-with-a-handmade-sliding-window-view-946c4acb98b4)

## Dataset

- [Shapes Dataset](https://www.kaggle.com/datasets/smeschke/four-shapes)

---
