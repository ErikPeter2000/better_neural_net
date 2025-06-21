use std::fs::File;
use std::io::{ Error, ErrorKind, Result };

use byteorder::{ BigEndian, ReadBytesExt };
use find_folder::Search;

use better_neural_net_core::matrix::Matrix;

const MNIST_MAGIC_SAMPLES: u32 = 2051;
const MNIST_MAGIC_LABELS: u32 = 2049;
const TRAINING_IMAGE_FILENAME: &str = "train-images-idx3-ubyte";
const TRAINING_LABEL_FILENAME: &str = "train-labels-idx1-ubyte";
const TEST_IMAGE_FILENAME: &str = "t10k-images-idx3-ubyte";
const TEST_LABEL_FILENAME: &str = "t10k-labels-idx1-ubyte";

/// Represents the MNIST dataset, containing training and test data.
pub struct MNISTData {
    pub training_data: Vec<(Matrix<f32>, Matrix<f32>)>,
    pub test_data: Vec<(Matrix<f32>, Matrix<f32>)>,
}

/// Reads the MNIST dataset from the expected location in the project directory.
///
/// The dataset is expected to be in `dataset/mnist` relative to the project root.
pub fn read_mnist_data() -> Result<MNISTData> {
    // Locate the mnist directory
    let dataset_directory = Search::ParentsThenKids(3, 3)
        .for_folder("dataset")
        .map_err(|_| { Error::new(ErrorKind::NotFound, "Could not find the `dataset` folder.") })?;
    let mnist_path = dataset_directory.join("mnist");
    if !mnist_path.exists() {
        return Err(
            Error::new(ErrorKind::NotFound, "MNIST dataset not found in the expected location.")
        );
    }

    // Construct paths for the MNIST files
    let training_images_path = mnist_path.join(TRAINING_IMAGE_FILENAME);
    let training_labels_path = mnist_path.join(TRAINING_LABEL_FILENAME);
    let test_images_path = mnist_path.join(TEST_IMAGE_FILENAME);
    let test_labels_path = mnist_path.join(TEST_LABEL_FILENAME);

    // Read the MNIST dataset files
    let training_images = read_mnist_image(training_images_path.to_str().unwrap())?;
    let training_labels = read_mnist_label(training_labels_path.to_str().unwrap())?;
    let test_images = read_mnist_image(test_images_path.to_str().unwrap())?;
    let test_labels = read_mnist_label(test_labels_path.to_str().unwrap())?;

    // Ensure that the number of images matches the number of labels
    if training_images.len() != training_labels.len() || test_images.len() != test_labels.len() {
        return Err(
            Error::new(ErrorKind::InvalidData, "Training and test data sizes do not match.")
        );
    }

    // Combine images and labels into tuples for training and test data
    let training_data = training_images.into_iter().zip(training_labels.into_iter()).collect();
    let test_data = test_images.into_iter().zip(test_labels.into_iter()).collect();

    Ok(MNISTData {
        training_data,
        test_data,
    })
}

/// Reads MNIST image samples from a file.
///
/// The magic number of the images should be `2051` as a 32-bit unsigned integer.`
fn read_mnist_image(file_path: &str) -> Result<Vec<Matrix<f32>>> {
    // Create a file reader
    let file = File::open(file_path)?;
    let mut reader = std::io::BufReader::new(file);

    // Validate magic number
    let magic_read = reader.read_u32::<BigEndian>()?;
    if magic_read != MNIST_MAGIC_SAMPLES {
        return Err(
            Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Invalid magic number: expected {}, got {}",
                    MNIST_MAGIC_SAMPLES,
                    magic_read
                )
            )
        );
    }

    // Read properties of the dataset
    let image_count = reader.read_u32::<BigEndian>()? as usize;
    let image_height = reader.read_u32::<BigEndian>()? as usize;
    let image_width = reader.read_u32::<BigEndian>()? as usize;
    let image_size = image_height * image_width;

    // Read images
    let mut images = Vec::with_capacity(image_count);
    for _ in 0..image_count {
        let mut data = vec![0.0; image_size];
        for pixel in data.iter_mut() {
            *pixel = (reader.read_u8()? as f32) / 255.0;
        }
        images.push(Matrix::new(image_height * image_width, 1, data));
    }

    Ok(images)
}

/// Reads MNIST labels from a file.
///
/// The magic number of the labels should be `2049` as a 32-bit unsigned integer.
fn read_mnist_label(file_path: &str) -> Result<Vec<Matrix<f32>>> {
    // Create a file reader
    let file = File::open(file_path)?;
    let mut reader = std::io::BufReader::new(file);

    // Validate magic number
    let magic_read = reader.read_u32::<BigEndian>()?;
    if magic_read != MNIST_MAGIC_LABELS {
        return Err(
            Error::new(
                ErrorKind::InvalidData,
                format!("Invalid magic number: expected {}, got {}", MNIST_MAGIC_LABELS, magic_read)
            )
        );
    }

    // Read properties of the dataset
    let label_count = reader.read_u32::<BigEndian>()? as usize;

    // Read labels
    let mut labels = Vec::with_capacity(label_count);
    for _ in 0..label_count {
        let label = reader.read_u8()? as usize;
        let mut data = vec![0.0; 10];
        data[label] = 1.0;
        labels.push(Matrix::new(10, 1, data));
    }

    Ok(labels)
}
