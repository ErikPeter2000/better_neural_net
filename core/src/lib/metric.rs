use std::fmt::{Debug, Display};

use crate::{ matrix::Matrix, network::Network };

/// Trait for defining a metric that can be used to evaluate model performance.
pub trait Metric: Display + Debug {}

/// Trait for evaluating a model's performance on a dataset.
pub trait MetricEvaluator {

    /// Evaluates the model on the provided dataset.
    ///
    /// Returns a concrete metric implementation.
    fn evaluate(&self, network: &Network, dataset: &[(Matrix<f32>, Matrix<f32>)]) -> Box<dyn Metric>;
}

/// Evaluator for accuracy metric.
#[derive(Debug, Clone)]
pub struct AccuracyEvaluator {}

/// Metric that represents the accuracy of a model's predictions.
#[derive(Debug, Clone)]
pub struct Accuracy {
    pub accuracy: f32,
}

impl Metric for Accuracy {}

impl Display for Accuracy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Accuracy: {:.2}%", self.accuracy * 100.0)
    }
}

impl AccuracyEvaluator {
    fn argmax(&self, matrix: &Matrix<f32>) -> usize {
        matrix.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map_or(0, |(i, _)| i)
    }
}

impl MetricEvaluator for AccuracyEvaluator {

    fn evaluate(&self, network: &Network, dataset: &[(Matrix<f32>, Matrix<f32>)]) -> Box<dyn Metric>
    {
        let dataset = dataset.to_vec();
        let mut correct_predictions = 0;
        let total_samples = dataset.len() as f32;

        for (input, target) in dataset {
            let output = network.forward(&input);
            let predicted_label = self.argmax(&output);
            let actual_label = self.argmax(&target);
            if predicted_label == actual_label {
                correct_predictions += 1;
            }
        }

        Box::new(Accuracy {
            accuracy: (correct_predictions as f32) / total_samples,
        })
    }
}
