# Better Neural Net

A from-scratch implementation of neural networks in Rust that prioritizes flexibility, performance, and educational value.

## Overview

Better Neural Net is a modular neural network library written in Rust that provides both a core library and a command-line interface. The project implements core neural network components including matrices, layers, activation functions, loss functions, and training algorithms from first principles.

## Project Structure

The project is organized as a Rust workspace with two main crates:

- `core`: The core neural network library with all the fundamental components
- `cli`: Command-line applications that use the core library, including a MNIST digit recognition example

## Features

### Core Library

- **Matrix Operations**: A custom matrix implementation with support for various operations
- **Neural Network Layers**: Fully-connected (dense) layer implementation
- **Activation Functions**: 
  - Sigmoid
  - ReLU
  - Tanh
  - Linear
- **Loss Functions**: 
  - Cross-Entropy Loss
- **Network Architecture**: Builder pattern for easy network construction
- **Training**: Flexible trainer with support for:
  - Batch training
  - Validation metrics
  - Training callbacks
  - Configurable hyperparameters

### CLI Applications

- The CLI project provides a command-line example for training a neural network on the MNIST dataset, demonstrating the core library's capabilities.

## Getting Started

### Prerequisites

- Rust 1.77 or later (2024 edition)
- Cargo package manager

### Installation

1. Clone the repository:
```bash
git clone https://your-repository-url.git
cd better_neural_net
```

2. Place the MNIST dataset in an `mnist` directory within `dataset`. The dataset should be in the format expected by the MNIST example (e.g., `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.).

3. Build the project:
```bash
cargo build --release
```

### Running the CLI Example

Execute the CLI example with:

```bash
cargo run --package better_neural_net_cli --release
```

## Usage

### Creating a Neural Network

```rust
let mut network = NetworkBuilder::new()
    .with_input_layer(
        Layer::new_random(784, 128, activation_function::ReLU, Some("Input Layer"))
    )
    .with_output_layer(
        Layer::new_random(128, 10, activation_function::Sigmoid, Some("Output Layer"))
    )
    .with_loss_function(loss_function::CrossEntropyLoss)
    .build()
    .expect("Failed to build the network");
```

### Training a Network

```rust
let mut trainer = TrainerBuilder::new()
    .with_network(&mut network)
    .with_epochs(100)
    .with_learning_rate(0.01)
    .with_data(training_data)
    .with_batch_size(32)
    .with_validation_data(validation_data)
    .with_metric_evaluators(vec![AccuracyEvaluator {}])
    .build()
    .expect("Failed to build the trainer");

// Train the network with a callback that prints information to the command line
let result = trainer.train(Some(|params| {
    println!("Epoch {}: Loss: {:.4}", params.epoch, params.av_loss);
})).unwrap();
```

### Evaluating a Network

```rust
let evaluator = AccuracyEvaluator {};
let metric = evaluator.evaluate(&network, &test_data);
println!("Test accuracy: {}", metric);
```

## Customization

### Adding New Activation Functions

Implement the `ActivationFunction` trait:

```rust
#[derive(Debug, Clone)]
pub struct NewActivation;

impl ActivationFunction for NewActivation {
    fn activate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        // Implementation
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        // Implementation
    }

    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}
```

### Adding New Loss Functions

Implement the `LossFunction` trait:

```rust
#[derive(Debug, Clone)]
pub struct NewLoss;

impl LossFunction for NewLoss {
    fn compute_loss(&self, predictions: &Matrix<f32>, targets: &Matrix<f32>) -> f32 {
        // Implementation
    }

    fn compute_gradient(&self, predictions: &Matrix<f32>, targets: &Matrix<f32>) -> Matrix<f32> {
        // Implementation
    }

    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(self.clone())
    }
}
```

## Performance Considerations

- Batch size can significantly impact training speed and memory usage
- The network architecture (number and size of layers) affects both performance and accuracy
- Matrix operations are implemented using naive algorithms. Therefore, this program is unsuitable for production use
- Samples are expected to be in column vector format to align with mathematical convention. This may differ from more optimal implementations that use row-major order

## Future Improvements

- Convolutional layers
- Dropout and other regularization techniques
- Learning rate scheduling
- Model serialization/deserialization
- GPU acceleration
- Additional activation functions
- Additional loss functions
- More examples and benchmarks

---

*This project was created for educational purposes and to demonstrate neural network implementation in Rust.*