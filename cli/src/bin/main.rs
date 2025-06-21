use better_neural_net_core::trainer::TrainingResult;
use indicatif::ProgressBar;

use better_neural_net_core::metric::MetricEvaluator;
use better_neural_net_core::trainer::EpochResult;
use better_neural_net_core::activation_function;
use better_neural_net_core::loss_function;
use better_neural_net_core::layer::Layer;
use better_neural_net_core::network::Network;
use better_neural_net_core::network::NetworkBuilder;
use better_neural_net_core::trainer::TrainerBuilder;
use better_neural_net_core::metric::AccuracyEvaluator;
use better_neural_net_core::matrix::Matrix;

use better_neural_net_cli::mnist;

const BATCH_SIZE: usize = 1;
const EPOCHS: usize = 100;
const LEARNING_RATE: f32 = 0.01;
const TRAINING_DATA_LIMIT: usize = 60000;
const VALIDATION_SPLIT: f32 = 0.1;

pub fn main() {
    // Load the MNIST dataset
    let (training_data, test_data) = load_dataset();

    // Split the training data into training and validation sets
    let (training_data, validation_data) = split_dataset(training_data, VALIDATION_SPLIT);

    // Create the neural network and trainer
    let mut network = create_network();
    let mut trainer = create_trainer(&mut network, training_data, validation_data);

    // Create a progress bar for tracking training progress
    let (progress_bar, progress_bar_clone) = create_progress_bar(EPOCHS);

    // Train the network and update the progress bar
    let training_result = train(&mut trainer, progress_bar_clone);
    progress_bar.finish_with_message("Training complete");

    // Evaluate the trained network on the test dataset
    evaluate(&network, training_result, test_data);
}

/// Loads the MNIST dataset.
fn load_dataset() -> (Vec<(Matrix<f32>, Matrix<f32>)>, Vec<(Matrix<f32>, Matrix<f32>)>) {
    let data = mnist::read_mnist_data().expect("Failed to load MNIST data");
    println!(
        "Loaded data. {} training samples, {} test samples.",
        data.training_data.len(),
        data.test_data.len()
    );
    (data.training_data, data.test_data)
}

/// Splits the dataset into training and validation sets based on the specified validation split ratio.
fn split_dataset(
    dataset: Vec<(Matrix<f32>, Matrix<f32>)>,
    validation_split: f32
) -> (Vec<(Matrix<f32>, Matrix<f32>)>, Vec<(Matrix<f32>, Matrix<f32>)>) {
    let split_index = (dataset.len() as f32 * (1.0 - validation_split)) as usize;
    let training_data = dataset[..split_index].to_vec();
    let validation_data = dataset[split_index..].to_vec();
    (training_data, validation_data)
}

// Creates a neural network with an input layer and an output layer for recognizing MNIST digits.
fn create_network() -> Network {
    NetworkBuilder::new()
        .with_input_layer(
            Layer::new_random(784, 32, activation_function::Sigmoid, Some("Input Layer"))
        )
        .with_output_layer(
            Layer::new_random(32, 10, activation_function::Sigmoid, Some("Output Layer"))
        )
        .with_loss_function(loss_function::CrossEntropyLoss)
        .build()
        .expect("Failed to build the network")
}

/// Creates a trainer for the neural network with the specified training data.
fn create_trainer(
    network: &mut Network,
    training_data: Vec<(Matrix<f32>, Matrix<f32>)>,
    validation_data: Vec<(Matrix<f32>, Matrix<f32>)>
) -> better_neural_net_core::trainer::Trainer {
    TrainerBuilder::new()
        .with_network(network)
        .with_epochs(EPOCHS)
        .with_learning_rate(LEARNING_RATE)
        .with_data(training_data)
        .with_data_limit(TRAINING_DATA_LIMIT)
        .with_batch_size(BATCH_SIZE)
        .with_validation_data(validation_data)
        .with_metric_evaluators(
            vec![AccuracyEvaluator {}]
        )
        .build()
        .expect("Failed to build the trainer")
}

/// Creates progress bar references for tracking training progress.
fn create_progress_bar(
    epochs: usize
) -> (std::sync::Arc<ProgressBar>, std::sync::Arc<ProgressBar>) {
    let progress_bar = ProgressBar::new(epochs as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle
            ::default_bar()
            .template("[{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("##-")
    );

    let arc = std::sync::Arc::new(progress_bar);
    let clone = std::sync::Arc::clone(&arc);
    (arc, clone)
}

/// Trains the network.
fn train(
    trainer: &mut better_neural_net_core::trainer::Trainer,
    progress_bar: std::sync::Arc<ProgressBar>
) -> TrainingResult {
    let result = trainer
        .train(
            Some(move |params: &EpochResult| {
                progress_bar.set_position(params.epoch as u64);
                progress_bar.set_message(
                    format!("{}", params)
                );
            })
        )
        .unwrap();

    result
}

/// Evaluates the trained network on the test dataset and prints the results.
fn evaluate(
    network: &Network,
    training_result: TrainingResult,
    test_data: Vec<(Matrix<f32>, Matrix<f32>)>
) {
    let evaluator = AccuracyEvaluator {};
    let metric = evaluator.evaluate(network, &test_data);

    println!("Final Loss: {:.4}", training_result.last_epoch_unchecked().av_loss);
    println!("{}", metric);

}