use std::fmt::Debug;

use crate::matrix::Matrix;

pub trait LossFunction: Debug {
    /// Computes the loss between the predicted and actual values.
    fn compute_loss(&self, predicted: &Matrix<f32>, actual: &Matrix<f32>) -> f32;

    /// Computes the gradient of the loss with respect to the predicted values.
    fn compute_gradient(&self, predicted: &Matrix<f32>, actual: &Matrix<f32>) -> Matrix<f32>;

    /// Clones the loss function and returns it as a boxed trait object.
    fn clone_box(&self) -> Box<dyn LossFunction>;
}

/// Mean Squared Error (MSE) loss function.
#[derive(Debug, Clone)]
pub struct MeanSquaredError;

/// Cross-Entropy Loss function, typically used for classification tasks.
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss;

impl LossFunction for MeanSquaredError {
    fn compute_loss(&self, predicted: &Matrix<f32>, actual: &Matrix<f32>) -> f32 {
        let diff = predicted.sub(actual);
        let squared_diff = diff.element_wise_multiply(&diff);
        squared_diff.iter().copied().sum::<f32>() / (predicted.cols()) as f32
    }

    fn compute_gradient(&self, predicted: &Matrix<f32>, actual: &Matrix<f32>) -> Matrix<f32> {
        let diff = predicted.sub(actual);
        diff.mul_scalar(2.0 / (predicted.cols()) as f32)
    }

    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(self.clone())
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, predicted: &Matrix<f32>, actual: &Matrix<f32>) -> f32 {
        let batch_size = predicted.cols() as f32;
        let mut loss = 0.0;
        for col in 0..predicted.cols() {
            for row in 0..predicted.rows() {
                let p = predicted[(row, col)];
                let t = actual[(row, col)];
                loss -= t * p.ln(); // Assuming p > 0; add small epsilon in practice
            }
        }
        loss / batch_size
    }

    fn compute_gradient(&self, predicted: &Matrix<f32>, actual: &Matrix<f32>) -> Matrix<f32> {
        predicted.sub(actual)
    }

    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(self.clone())
    }
}