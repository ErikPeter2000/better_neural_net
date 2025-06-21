use std::fmt::Display;

use rand::seq::SliceRandom;

use crate::layer::ForwardLayerPassResult;
use crate::matrix::Matrix;
use crate::metric::{ Metric, MetricEvaluator };
use crate::network::Network;

/// Represents the history of training epochs.
///
/// The `TrainingHistory` struct contains a vector of `EpochResult` instances,
#[derive(Debug)]
pub struct TrainingResult {
    pub epoch_history: Vec<EpochResult>,
}

/// Parameters for the epoch callback function.
///
/// Contains data about the epoch that was just completed.
#[derive(Debug)]
pub struct EpochResult {
    /// The current epoch number.
    pub epoch: usize,
    /// The loss value calculated for the epoch.
    pub av_loss: f32,
    /// A list of metrics calculated for the epoch.
    pub metrics: Vec<Box<dyn Metric>>,
}

/// Represents the result of a forward pass through the network.
#[derive(Debug)]
pub struct ForwardPassResult {
    layer_results: Vec<ForwardLayerPassResult>,
    output: Matrix<f32>,
}

/// Builder for creating a `Trainer` instance.
///
/// The `TrainerBuilder` allows you to specify training parameters such as the network,
/// number of epochs, learning rate, training data, and whether to shuffle the data before training.
///
/// # Example
/// ```rust
/// use better_neural_net::network::Network;
///
/// let trainer = TrainerBuilder::new()
///    .with_network(&mut network)
///   .with_epochs(100)
///   .with_learning_rate(0.01)
///   .with_data(training_data)
///   .with_shuffle(true)
///   .build()
///   .unwrap();
///
/// trainer.train(Some(|params| { println!("{:?}", params); }));
/// ```
pub struct TrainerBuilder<'a> {
    network: Option<&'a mut Network>,
    epochs: Option<usize>,
    learning_rate: Option<f32>,
    training_data: Option<Vec<(Matrix<f32>, Matrix<f32>)>>,
    training_data_limit: Option<usize>,
    validation_data: Option<Vec<(Matrix<f32>, Matrix<f32>)>>,
    batch_size: Option<usize>,
    metric_evaluators: Vec<Box<dyn MetricEvaluator>>,
}

/// Used to train a neural network.
///
/// To create a `Trainer`, see `TrainerBuilder`.
pub struct Trainer<'a> {
    network: &'a mut Network,
    epochs: usize,
    learning_rate: f32,
    training_data: Option<Vec<(Matrix<f32>, Matrix<f32>)>>,
    batch_size: Option<usize>,
    metric_evaluators: Vec<Box<dyn MetricEvaluator>>,
    validation_data: Option<Vec<(Matrix<f32>, Matrix<f32>)>>,
}

impl TrainingResult {
    /// Returns the number of epochs in the training history.
    pub fn last_epoch(&self) -> Result<&EpochResult, String> {
        self.epoch_history.last().ok_or(
            "No epochs recorded in training history".to_string()
        )
    }

    /// Returns the last epoch result without checking for existence.
    pub fn last_epoch_unchecked(&self) -> &EpochResult {
        self.epoch_history.last().expect("No epochs recorded in training history")
    }
}

impl EpochResult {
    /// Returns the average loss for the epoch.
    pub fn av_loss(&self) -> f32 {
        self.av_loss
    }

    /// Returns the epoch number.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Returns the metrics calculated for the epoch.
    pub fn metrics(&self) -> &Vec<Box<dyn Metric>> {
        &self.metrics
    }

    /// Returns a string representation of the metrics for display.
    pub fn metrics_display(&self) -> String {
        self.metrics
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Display for EpochResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Epoch {}: Loss: {:.4} {}", self.epoch, self.av_loss, self.metrics_display())
    }
}

impl<'a> TrainerBuilder<'a> {
    /// Creates a new `TrainerBuilder` instance.
    pub fn new() -> TrainerBuilder<'a> {
        TrainerBuilder {
            network: None,
            epochs: None,
            learning_rate: None,
            training_data: None,
            training_data_limit: None,
            batch_size: None,
            metric_evaluators: Vec::new(),
            validation_data: None,
        }
    }

    /// Sets the network to be trained.
    pub fn with_network(mut self, network: &'a mut Network) -> TrainerBuilder<'a> {
        self.network = Some(network);
        self
    }

    /// Sets the number of epochs for training.
    pub fn with_epochs(mut self, epochs: usize) -> TrainerBuilder<'a> {
        self.epochs = Some(epochs);
        self
    }

    /// Sets the learning rate for training.
    pub fn with_learning_rate(mut self, learning_rate: f32) -> TrainerBuilder<'a> {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Sets the training data for the trainer.
    pub fn with_data(mut self, data: Vec<(Matrix<f32>, Matrix<f32>)>) -> TrainerBuilder<'a> {
        self.training_data = Some(data);
        self
    }

    /// Sets the batch size for training.
    pub fn with_batch_size(mut self, batch_size: usize) -> TrainerBuilder<'a> {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets the limit on the number of training samples.
    pub fn with_data_limit(mut self, limit: usize) -> TrainerBuilder<'a> {
        self.training_data_limit = Some(limit);
        self
    }

    /// Adds metric evaluators to the trainer.
    pub fn with_metric_evaluators<E, I>(mut self, metric_evaluators: I) -> TrainerBuilder<'a>
        where E: MetricEvaluator + 'static, I: IntoIterator<Item = E>
    {
        self.metric_evaluators = metric_evaluators
            .into_iter()
            .map(|e| Box::new(e) as Box<dyn MetricEvaluator>)
            .collect();
        self
    }

    /// Sets the validation data for the trainer.
    pub fn with_validation_data(
        mut self,
        validation_data: Vec<(Matrix<f32>, Matrix<f32>)>
    ) -> TrainerBuilder<'a> {
        self.validation_data = Some(validation_data);
        self
    }

    /// Builds the `Trainer` instance.
    ///
    /// # Errors
    /// Returns an error if any required parameters are unspecified:
    /// - `network`
    /// - `epochs`
    /// - `learning_rate`
    /// - `training_data`
    /// - `validation_data`, only if `metric_evaluators` are provided
    pub fn build(mut self) -> Result<Trainer<'a>, String> {
        // Check provided configuration is sufficient.
        if self.network.is_none() {
            return Err("Network is required".to_string());
        }
        if self.epochs.is_none() {
            return Err("Number of epochs is required".to_string());
        }
        if self.learning_rate.is_none() {
            return Err("Learning rate is required".to_string());
        }
        if self.training_data.is_none() {
            return Err("Training data is required".to_string());
        }
        if !self.metric_evaluators.is_empty() && self.validation_data.is_none() {
            return Err("Validation data is required if metric evaluators are provided".to_string());
        }

        // Take ownership of the data from self
        let mut data = self.training_data.take().unwrap();

        // Limit the data if a limit is specified
        if let Some(limit) = self.training_data_limit {
            if limit < data.len() {
                data.truncate(limit);
            }
        }

        Ok(Trainer {
            network: self.network.unwrap(),
            epochs: self.epochs.unwrap(),
            learning_rate: self.learning_rate.unwrap(),
            training_data: Some(data),
            validation_data: self.validation_data.take(),
            batch_size: self.batch_size,
            metric_evaluators: self.metric_evaluators,
        })
    }
}

impl<'a> Trainer<'a> {
    /// Trains the neural network for the specified number of epochs.
    ///
    /// # Parameters
    /// - `epoch_callback`: An optional callback function that is called after each epoch.
    /// The callback receives an `EpochResult` struct containing the current epoch number and
    /// the loss value for that epoch.
    ///
    /// # Returns
    /// Returns a `TrainingHistory` struct containing the history of epochs,
    pub fn train<F>(&mut self, epoch_callback: Option<F>) -> Result<TrainingResult, String>
        where F: Fn(&EpochResult) + 'static
    {
        let mut data = self.training_data.take().ok_or("No training data provided")?;

        let mut training_history = TrainingResult {
            epoch_history: vec!(EpochResult { epoch: 0, av_loss: 0.0, metrics: self.evaluate_metrics()? }),
        };

        // Train the network for the specified number of epochs.
        for i in 0..self.epochs {
            // Shuffle the data and create batches.
            data.shuffle(&mut rand::rng());
            let batched_data = self.batch_data(&data);
            let metrics = self.evaluate_metrics()?;

            // Calculate loss
            let loss = batched_data
                .iter()
                .map(|(input, target)| self.training_pass(input, target))
                .sum::<f32>();

            // Epoch result
            let epoch_result = EpochResult {
                epoch: i,
                av_loss: loss / batched_data.len() as f32,
                metrics,
            };

            // If a callback is provided, call it with the epoch result.
            if let Some(callback) = &epoch_callback {
                callback(&epoch_result);
            }

            training_history.epoch_history.push(epoch_result);
        }

        // Restore the data to the trainer after training.
        self.training_data = Some(data);

        Ok(training_history)
    }

    /// Returns the number of epochs configured for training.
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the learning rate configured for training.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the network being trained.
    pub fn network(&self) -> &Network {
        self.network
    }

    /// Performs a single training pass through the network.
    fn training_pass(&mut self, input: &Matrix<f32>, target: &Matrix<f32>) -> f32 {
        // Forward pass
        let forward_result = self.forward_pass(input);

        // Compute loss
        let loss = self.network.loss_function().compute_loss(&forward_result.output, target);

        // Backward pass
        self.backward_pass(&forward_result, input, target); // Fixed parameter order

        loss
    }

    /// Performs a forward pass through the network.
    ///
    /// The result contains the activations of each layer and the final output,
    /// to be used in backpropagation.
    fn forward_pass(&self, input: &Matrix<f32>) -> ForwardPassResult {
        // Initialize a vector to hold the results of each layer's forward pass.
        let mut layer_results = Vec::<ForwardLayerPassResult>::new();
        let mut last_activations = input.clone();

        // Iterate through each layer in the network and perform the forward pass.
        for layer in self.network.layers() {
            let layer_result = layer.forward(&last_activations);
            last_activations = layer_result.activations().clone();
            layer_results.push(layer_result);
        }

        ForwardPassResult {
            layer_results,
            output: last_activations,
        }
    }

    /// Performs a backward backpropagation pass through the network.
    ///
    /// This method computes the gradients for each layer based on the loss function
    /// and updates the weights and biases accordingly.
    fn backward_pass(
        &mut self,
        forward_result: &ForwardPassResult,
        network_input: &Matrix<f32>,
        target: &Matrix<f32>
    ) {
        // Get relevant data for simpler access.
        let layer_results = &forward_result.layer_results;
        let layer_count = layer_results.len();

        // Start with the loss derivative from the output layer
        // This will be used as the "chain" for backpropagation.
        let mut loss_derivative = self.network
            .loss_function()
            .compute_gradient(&forward_result.output, target);

        // Iterate through the layers in reverse order to perform backpropagation.
        for (i, layer) in self.network.layers_mut().iter_mut().rev().enumerate() {
            let forward_layer_result = &layer_results[layer_count - 1 - i];
            // If this is the first layer in the network, use the original input.
            let layer_input = if i == layer_count - 1 {
                network_input
            } else {
                layer_results[layer_count - 2 - i].activations()
            };

            // Perform the backward pass for the current layer.
            let backward_result = layer.backward(
                forward_layer_result,
                &loss_derivative,
                &layer_input
            );

            // Update the loss derivative for the next layer in the chain.
            loss_derivative = backward_result.input_derivative().clone();
            layer.update_weights(
                &backward_result.weights_derivative(),
                &backward_result.biases_derivative(),
                self.learning_rate
            );
        }
    }

    /// Splits given data into batches of the specified size.
    ///
    /// This is done by turning each chunk of vectors into a single matrix.
    fn batch_data(&self, data: &[(Matrix<f32>, Matrix<f32>)]) -> Vec<(Matrix<f32>, Matrix<f32>)> {
        data.chunks(self.batch_size.unwrap_or(1))
            .map(|batch| {
                // Unzip samples and inputs
                let (inputs, targets): (Vec<_>, Vec<_>) = batch.iter().cloned().unzip();

                // Turn inputs into a matrix where each column is an input sample
                let input_matrix = Matrix::new(
                    inputs[0].rows(),
                    inputs.len(),
                    inputs
                        .iter()
                        .flat_map(|m| m.iter().copied())
                        .collect::<Vec<_>>()
                );

                // Do similarly to targets
                let target_matrix = Matrix::new(
                    targets[0].rows(),
                    targets.len(),
                    targets
                        .iter()
                        .flat_map(|m| m.iter().copied())
                        .collect::<Vec<_>>()
                );

                (input_matrix, target_matrix)
            })
            .collect()
    }

    /// Evaluates the model's performance on the validation dataset.
    fn evaluate_metrics(&self) -> Result<Vec<Box<dyn Metric>>, String> {
        if self.metric_evaluators.is_empty() {
            Ok(vec![])
        } else {
            let validation_data = self.validation_data
                .as_ref()
                .ok_or("No validation data provided")?;
            let metrics = self.metric_evaluators
                .iter()
                .map(|evaluator| evaluator.evaluate(&self.network, validation_data))
                .collect::<Vec<_>>();
            Ok(metrics)
        }
    }
}
