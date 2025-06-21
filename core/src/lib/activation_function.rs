use std::fmt::Debug;

use crate::matrix::Matrix;

/// Represents an activation function that can be applied to a matrix of inputs.
///
/// It should support batching.
pub trait ActivationFunction: Debug {
    /// Applies the activation function to the input matrix.
    /// Batched processing is supported, meaning the function can handle matrices with multiple rows and columns.
    fn activate(&self, input: &Matrix<f32>) -> Matrix<f32>;

    /// Computes the derivative of the activation function with respect to the input matrix.
    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32>;

    /// Clones the activation function and returns it as a boxed trait object.
    fn clone_box(&self) -> Box<dyn ActivationFunction>;
}

/// Linear activation function.
///
/// The Linear activation function is the identity function.
#[derive(Debug, Clone)]
pub struct Linear;

/// Sigmoid activation function.
///
/// The Sigmoid function smoothly maps inputs into the range (0, 1).
/// f(x) = 1 / (1 + exp(-x))
/// f'(x) = f(x) * (1 - f(x)).
#[derive(Debug, Clone)]
pub struct Sigmoid;

/// ReLU (Rectified Linear Unit) activation function.
///
/// The ReLU function outputs the input if it is positive; otherwise, it outputs zero.
/// f(x) = max(0, x)
/// f'(x) = 1 if x > 0, else 0.
#[derive(Debug, Clone)]
pub struct ReLU;

/// Tanh activation function.
///
/// The Tanh function maps any real-valued number into the range (-1, 1).
/// f(x) = tanh(x)
/// f'(x) = 1 - f(x)^2.
#[derive(Debug, Clone)]
pub struct Tanh;

impl ActivationFunction for Linear {
    fn activate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        input.clone()
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        Matrix::from_value(input.rows(), input.cols(), 1.0)
    }

    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

impl ActivationFunction for Sigmoid {
    fn activate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        input.element_wise_map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        let activated = self.activate(input);
        activated.element_wise_map(|x| x * (1.0 - x))
    }

    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

impl ActivationFunction for ReLU {
    fn activate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        input.element_wise_map(|x| x.max(0.0))
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        input.element_wise_map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
    }

    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

impl ActivationFunction for Tanh {
    fn activate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        input.element_wise_map(|x| x.tanh())
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        let activated = self.activate(input);
        activated.element_wise_map(|x| 1.0 - x * x)
    }

    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}
