use std::fmt::Display;

use crate::layer::Layer;
use crate::loss_function::LossFunction;
use crate::matrix::Matrix;

/// Represents a neural network composed of multiple layers, and a loss function.
/// 
/// The `Network` struct encapsulates the layers of the neural network and the loss function used for training.
/// To train the network, you can use the `Trainer` struct, which is built using the `TrainerBuilder`.
#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    loss_function: Box<dyn LossFunction>,
}

/// A builder for constructing a neural network.
///
/// The `NetworkBuilder` allows you to specify the input layer, output layer, and hidden layers.
/// For input and output layers, you can either provide a specific layer or just the size.
/// If only the size is provided, a passthrough layer will be created.
///
/// # Example
/// ```rust
/// use better_neural_net::activation_function::{ ReLU, Softmax };
/// use better_neural_net::layer::Layer;
/// use better_neural_net::network::NetworkBuilder;
///
/// let network = NetworkBuilder::new()
///     .with_input_size(784)
///     .with_hidden_layers([
///         Layer::new_random(784, 128, ReLU, Some("Hidden Layer 1")),
///         Layer::new_random(128, 64, ReLU, Some("Hidden Layer 2")),
///     ])
///     .with_output_layer(Layer::new_random(64, 10, Softmax, Some("Output Layer")))
///     .build()
///     .unwrap();
/// ```
pub struct NetworkBuilder {
    hidden_layers: Vec<Layer>,
    input_layer: Option<Layer>,
    output_layer: Option<Layer>,
    input_size: Option<usize>,
    output_size: Option<usize>,
    loss_function: Option<Box<dyn LossFunction>>,
}

impl Network {
    /// Creates a new `Network` with the specified layers.
    /// 
    /// # Arguments
    /// - `layers`: A vector of `Layer` instances that make up the network.
    /// - `loss_function`: The loss function to be used by the network.
    pub fn new<L>(layers: Vec<Layer>, loss_function: L) -> Self where L: LossFunction + 'static {
        Network { layers, loss_function: Box::new(loss_function) }
    }

    /// Returns a reference to the layers of the network.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Returns a mutable reference to the layers of the network.
    pub fn layers_mut(&mut self) -> &mut Vec<Layer> {
        &mut self.layers
    }

    /// Returns the loss function used by the network.
    pub fn loss_function(&self) -> &Box<dyn LossFunction> {
        &self.loss_function
    }

    /// Performs a forward pass through the network.
    pub fn forward(&self, input: &Matrix<f32>) -> Matrix<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output).activations().clone();
        }
        output
    }
}

impl NetworkBuilder {
    /// Creates a new, empty `NetworkBuilder` instance.
    /// 
    /// Remember to specify at least the input layer or size, output layer or size, and a loss function before building the network.
    pub fn new() -> Self {
        NetworkBuilder {
            hidden_layers: Vec::new(),
            input_layer: None,
            output_layer: None,
            input_size: None,
            output_size: None,
            loss_function: None,
        }
    }

    /// Sets the input layer of the network.
    pub fn with_input_layer(mut self, layer: Layer) -> Self {
        self.input_size = Some(layer.input_size());
        self.input_layer = Some(layer);
        self
    }

    /// Sets the input size of the network. The input layer will be created as a passthrough layer if not specified.
    pub fn with_input_size(mut self, size: usize) -> Self {
        self.input_size = Some(size);
        self
    }

    /// Adds an output layer to the network.
    pub fn with_output_layer(mut self, layer: Layer) -> Self {
        self.output_size = Some(layer.output_size());
        self.output_layer = Some(layer);
        self
    }

    /// Sets the output size of the network. The output layer will be created as a passthrough layer if not specified.
    pub fn with_output_size(mut self, size: usize) -> Self {
        self.output_size = Some(size);
        self
    }

    /// Adds a hidden layer to the network.
    pub fn with_hidden_layers<I>(mut self, layers: I) -> Self where I: IntoIterator<Item = Layer> {
        self.hidden_layers.extend(layers);
        self
    }

    /// Sets the loss function for the network.
    pub fn with_loss_function<L>(mut self, loss_function: L) -> Self where L: LossFunction + 'static {
        self.loss_function = Some(Box::new(loss_function));
        self
    }

    /// Build the network with the specified layers and sizes.
    ///
    /// # Errors
    /// Returns an error if any of the following are unspecified:
    /// - `input_layer` or `input_size`
    /// - `output_layer` or `output_size`
    /// - `loss_function`
    pub fn build(self) -> Result<Network, String> {
        // Check provided configuration is sufficient.
        if self.input_size.is_none() {
            return Err("Input layer or size must be specified".to_string());
        }
        if self.output_size.is_none() {
            return Err("Output layer or size must be specified".to_string());
        }
        if self.loss_function.is_none() {
            return Err("Loss function must be specified".to_string());
        }

        let mut layers = Vec::<Layer>::new();

        // Input layer
        if let Some(input_layer) = self.input_layer {
            layers.push(input_layer);
        } else {
            let input_size = self.input_size.unwrap();
            let output_size = self.hidden_layers
                .first()
                .map_or(self.output_size.unwrap(), |l| l.input_size());
            layers.push(
                Layer::new_passthrough(input_size, output_size, Some("Input Layer".to_string()))
            );
        }

        // Hidden layers
        layers.extend(self.hidden_layers);

        // Output layer
        if let Some(output_layer) = self.output_layer {
            layers.push(output_layer);
        } else {
            let input_size = layers.last().map_or(self.input_size.unwrap(), |l| l.output_size());
            let output_size = self.output_size.unwrap();
            layers.push(
                Layer::new_passthrough(input_size, output_size, Some("Output Layer".to_string()))
            );
        }

        // Validate network structure and collect all errors
        let mut errors = String::new();
        for i in 0..layers.len() - 1 {
            let current_layer = &layers[i];
            let next_layer = &layers[i + 1];
            if current_layer.output_size() != next_layer.input_size() {
                errors.push_str(
                    &format!(
                        "Layer {} output size ({}) does not match layer {} input size ({}).\n",
                        i,
                        current_layer.output_size(),
                        i + 1,
                        next_layer.input_size()
                    )
                );
            }
        }

        // Return result
        if errors.is_empty() {
            Ok(Network { layers, loss_function: self.loss_function.unwrap() })
        } else {
            Err(errors)
        }
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Network {
            layers: self.layers.clone(),
            loss_function: self.loss_function.clone_box(),
        }
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Network with {} layers:\n", self.layers.len())?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "================ Layer {} ================ \n{}", i, layer)?;
        }
        Ok(())
    }
}
