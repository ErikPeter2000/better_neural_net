use std::fmt::{ Debug, Display };

use crate::matrix::Matrix;
use crate::activation_function::{ self, ActivationFunction };

/// Contains the results of a forward pass through a neural network layer.
///
/// `ForwardPassResult` holds both the `activations` (output after the activation function)
/// and the `pre_activations` (raw output before the activation function). These are
/// typically used during backpropagation.
#[derive(Debug, Clone)]
pub struct ForwardLayerPassResult {
    pre_activations: Matrix<f32>,
    activations: Matrix<f32>,
}

/// Contains the results of a backward pass through a neural network layer.
///
/// `BackwardPassResult` holds the `gradients` of the loss with respect to the pre-activations,
/// the `weight_gradients` (derivative of the loss with respect to the weights), and
/// the `bias_gradients` (derivative of the loss with respect to the biases).
#[derive(Debug, Clone)]
pub struct BackwardLayerPassResult {
    input_derivative: Matrix<f32>,
    weights_derivative: Matrix<f32>,
    biases_derivative: Matrix<f32>,
}

/// Represents a single layer in a neural network.
///
/// A `Layer` consists of weights, biases, and an activation function. Currently, all layers are dense layers.
///
/// # Example
/// ```rust
/// use better_neural_net::activation_function::{ ReLU, Softmax };
/// use better_neural_net::layer::Layer;
/// use better_neural_net::matrix::Matrix;
///
/// let layer = Layer::new(
///     784, // input size
///     128, // output size
///     Matrix::from_value(128, 784, 0.1), // weights initialized to 0.1, for example
///     Matrix::from_value(128, 1, 0.0),   // biases initialized to zero
///     ReLU, // activation function
///     Some("Hidden Layer 1") // optional name
/// );
/// 
/// let random_layer = Layer::new_random(784, 128, ReLU, Some("Random Layer")); // creates a layer with random weights and biases
/// let passthrough_layer = Layer::new_passthrough(784, 128, Some("Passthrough Layer")); // creates a passthrough layer with identity weights and zero biases
/// ```
#[derive(Debug)]
pub struct Layer {
    name: String,
    input_size: usize,
    output_size: usize,
    weights: Matrix<f32>,
    biases: Matrix<f32>,
    activation_function: Box<dyn ActivationFunction>,
}

impl ForwardLayerPassResult {
    pub fn new(activations: Matrix<f32>, pre_activations: Matrix<f32>) -> Self {
        ForwardLayerPassResult {
            activations,
            pre_activations,
        }
    }

    /// Returns the activations (output after the activation function).
    pub fn activations(&self) -> &Matrix<f32> {
        &self.activations
    }

    /// Returns the pre-activations (raw output before the activation function).
    pub fn pre_activations(&self) -> &Matrix<f32> {
        &self.pre_activations
    }
}

impl BackwardLayerPassResult {
    pub fn new(
        input_derivative: Matrix<f32>,
        weights_derivative: Matrix<f32>,
        biases_derivative: Matrix<f32>
    ) -> Self {
        BackwardLayerPassResult {
            input_derivative,
            weights_derivative,
            biases_derivative,
        }
    }

    /// Returns the derivative of the loss with respect to the input to this layer.
    pub fn input_derivative(&self) -> &Matrix<f32> {
        &self.input_derivative
    }

    /// Returns the derivative of the loss with respect to the weights.
    pub fn weights_derivative(&self) -> &Matrix<f32> {
        &self.weights_derivative
    }

    /// Returns the derivative of the loss with respect to the biases.
    pub fn biases_derivative(&self) -> &Matrix<f32> {
        &self.biases_derivative
    }
}

impl Layer {
    /// Creates a new layer with the specified input size, output size, and activation function.
    ///
    /// # Panics
    /// Panics if the dimensions of the weights and biases do not match the specified input and output sizes, or if the biases are not a column vector.
    ///
    /// # Parameters
    /// - `input_size`: The number of inputs to the layer.
    /// - `output_size`: The number of outputs from the layer.
    /// - `weights`: A matrix representing the weights of the layer.
    /// - `biases`: A matrix representing the biases of the layer.
    /// - `activation_function`: The activation function to be applied to the layer's output.
    /// - `name`: An optional name for the layer, which defaults `Layer {input_size}->{output_size}` if not provided.
    pub fn new<A, S>(
        input_size: usize,
        output_size: usize,
        weights: Matrix<f32>,
        biases: Matrix<f32>,
        activation_function: A,
        name: Option<S>
    )
        -> Self
        where A: ActivationFunction + 'static, S: Into<String>
    {
        assert_eq!(weights.rows(), output_size, "Weights rows must match output size");
        assert_eq!(weights.cols(), input_size, "Weights columns must match input size");
        assert_eq!(biases.rows(), output_size, "Biases rows must match output size");
        assert_eq!(biases.cols(), 1, "Biases must be a column vector");
        assert_eq!(
            weights.rows(),
            biases.rows(),
            "Weights and biases must have the same number of rows"
        );
        Layer {
            input_size,
            output_size,
            weights,
            biases,
            activation_function: Box::new(activation_function),
            name: name.map_or_else(
                || format!("Layer {}->{}", input_size, output_size),
                |n| n.into()
            ),
        }
    }

    /// Creates a new layer with random weights and biases, using the specified input size, output size, and activation function.
    ///
    /// Weights and biases are initialized with random values in the range [-1, 1].
    /// The activation function is provided as an optional parameter, defaulting to a linear activation if unspecified.
    pub fn new_random<A, S>(
        input_size: usize,
        output_size: usize,
        activation_function: A,
        name: Option<S>
    )
        -> Self
        where A: ActivationFunction + 'static, S: Into<String>
    {
        let weights = Matrix::new(
            output_size,
            input_size,
            (0..output_size * input_size).map(|_| rand::random::<f32>() * 2.0 - 1.0)
        );
        let biases = Matrix::from_value(output_size, 1, 0.0);
        Layer::new(input_size, output_size, weights, biases, activation_function, name)
    }

    pub fn new_passthrough<S>(input_size: usize, output_size: usize, name: Option<S>) -> Self
        where S: Into<String>
    {
        let weights = Matrix::from_diagonal(output_size, input_size, 1.0, 0.0);
        let biases = Matrix::from_value(output_size, 1, 0.0);
        Layer::new(input_size, output_size, weights, biases, activation_function::Linear, name)
    }

    /// Returns the input size of the layer.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the output size of the layer.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns a reference to the weights of the layer.
    pub fn weights(&self) -> &Matrix<f32> {
        &self.weights
    }

    /// Returns a reference to the biases of the layer.
    pub fn biases(&self) -> &Matrix<f32> {
        &self.biases
    }

    /// Returns a reference to the activation function of the layer.
    pub fn activation_function(&self) -> &Box<dyn ActivationFunction> {
        &self.activation_function
    }

    /// Performs a forward pass through the layer, applying the weights, biases, and activation function.
    pub fn forward(&self, input: &Matrix<f32>) -> ForwardLayerPassResult {
        assert_eq!(input.rows(), self.input_size, "Input size must match layer input size");

        // Compute pre-activations: Z = W * X + b
        let pre_activations = self.weights.mul(input).add_column_vector(&self.biases);

        // Apply activation function
        let activations = self.activation_function.activate(&pre_activations);

        ForwardLayerPassResult::new(activations, pre_activations)
    }

    // Performs a backward pass through the layer, computing the gradients of the loss with respect to the inputs, weights, and biases.
    pub fn backward(
        &self,
        forward_pass_result: &ForwardLayerPassResult,
        loss_derivative: &Matrix<f32>,
        input: &Matrix<f32>
    ) -> BackwardLayerPassResult {
        let activation_function_derivative = &self.activation_function.derivative(&forward_pass_result.pre_activations);
        let loss_vs_pre_activation = loss_derivative.element_wise_multiply(&activation_function_derivative);
        let weights_deriv = loss_vs_pre_activation.mul(&input.transpose());
        let biases_deriv = loss_vs_pre_activation.clone().collapse_columns().mul_scalar(1.0 / loss_vs_pre_activation.cols() as f32);
        let input_deriv = self.weights.transpose().mul(&loss_vs_pre_activation);

        BackwardLayerPassResult::new(input_deriv, weights_deriv, biases_deriv)
    }

    /// Updates the weights and biases of the layer using the computed gradients and a learning rate.
    pub fn update_weights(
        &mut self,
        weights_derivative: &Matrix<f32>,
        biases_derivative: &Matrix<f32>,
        learning_rate: f32
    ) {
        assert_eq!(weights_derivative.rows(), self.output_size, "Weights derivative rows must match output size");
        assert_eq!(weights_derivative.cols(), self.input_size, "Weights derivative columns must match input size");
        assert_eq!(biases_derivative.rows(), self.output_size, "Biases derivative rows must match output size");
        assert_eq!(biases_derivative.cols(), 1, "Biases derivative must be a column vector");

        // Update weights and biases
        self.weights = self.weights.sub(&weights_derivative.mul_scalar(learning_rate));
        self.biases = self.biases.sub(&biases_derivative.mul_scalar(learning_rate));
    }
}

impl Clone for Layer {
    fn clone(&self) -> Self {
        Layer {
            name: self.name.clone(),
            input_size: self.input_size,
            output_size: self.output_size,
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            activation_function: self.activation_function.clone_box(),
        }
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Name: {}, \ninput_size: {}, \noutput_size: {}, \nactivation_function: {:?}",
            self.name,
            self.input_size,
            self.output_size,
            self.activation_function
        )
    }
}
