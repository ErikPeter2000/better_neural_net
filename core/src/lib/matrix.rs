use std::fmt::{ Debug, Display };
use std::ops::{ Add, Index, IndexMut, Mul, Neg, Sub };

/// A simple 2D matrix structure for storing data of type `T`.
///
/// # Example
/// ```rust
/// use better_neural_net::matrix::Matrix;
///
/// // Create a 2x2 matrix with the provided data
/// let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Creates a new `Matrix` with the specified number of rows, columns, and the provided data.
    pub fn new<I: IntoIterator<Item = T>>(rows: usize, cols: usize, data: I) -> Self
        where I::IntoIter: ExactSizeIterator
    {
        let data: Vec<T> = data.into_iter().collect();
        assert_eq!(rows * cols, data.len(), "Data length does not match dimensions");
        Matrix { rows, cols, data }
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns a reference to the underlying data of the matrix.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a row of the matrix as a slice.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn row(&self, index: usize) -> &[T] {
        assert!(index < self.rows, "Row index out of bounds");
        &self.data[index * self.cols..(index + 1) * self.cols]
    }

    // Returns a column of the matrix as a vector of references.
    pub fn col(&self, index: usize) -> Vec<&T> {
        assert!(index < self.cols, "Column index out of bounds");
        (0..self.rows).map(|row| &self.data[row * self.cols + index]).collect()
    }

    /// Returns an iterator over the elements of the matrix.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the elements of the matrix.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Applies an element-wise operation to two matrices of the same dimensions, to produce a new matrix.
    ///
    /// # Panics
    /// Panics if the dimensions of the matrices do not match.
    pub fn element_wise_op<F: Fn(&T, &T) -> T>(&self, other: &Matrix<T>, op: F) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix columns must match");

        let mut result_data = Vec::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            result_data.push(op(&self.data[i], &other.data[i]));
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

    /// Maps a function over the elements of the matrix, producing a new matrix.
    pub fn element_wise_map<F: Fn(&T) -> T>(&self, f: F) -> Matrix<T> {
        let mut mapped_data = Vec::with_capacity(self.rows * self.cols);
        for item in &self.data {
            mapped_data.push(f(item));
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: mapped_data,
        }
    }
}

impl<T> Matrix<T> where T: Clone {
    /// Returns a new matrix that is the transpose of the current matrix.
    pub fn transpose(&self) -> Matrix<T> {
        let mut result = Matrix::from_value(self.cols, self.rows, self.data[0].clone());
        for col in 0..self.cols {
            for row in 0..self.rows {
                result[(col, row)] = self[(row, col)].clone();
            }
        }
        result
    }

    /// Creates a new `Matrix` with the specified number of rows and columns, initialized with a default value.
    pub fn from_value(rows: usize, cols: usize, value: T) -> Self where T: Clone {
        let data = vec![value; rows * cols];
        Matrix { rows, cols, data }
    }

    /// Creates a new `Matrix` with the specified number of rows and columns, and a certain value across the diagonal.
    pub fn from_diagonal(rows: usize, cols: usize, diagonal_value: T, zero_value: T) -> Self
        where T: Clone
    {
        let mut data = vec![zero_value; rows * cols];
        for i in 0..rows.min(cols) {
            data[i * cols + i] = diagonal_value.clone();
        }
        Matrix { rows, cols, data }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        &mut self.data[row * self.cols + col]
    }
}

impl<T> Matrix<T> where T: Add<Output = T> + Copy {
    /// Adds two matrices element-wise and returns a new matrix.
    ///
    /// # Panics
    /// Panics if the dimensions of the matrices do not match.
    pub fn add(&self, other: &Matrix<T>) -> Matrix<T> {
        self.element_wise_op(other, |a, b| *a + *b)
    }

    /// Adds a column vector to each column of the this matrix.
    ///
    /// # Panics
    /// Panics if the input matrix is not a column vector or if the number of rows does not match.
    pub fn add_column_vector(&self, vector: &Matrix<T>) -> Matrix<T> {
        assert_eq!(vector.cols, 1, "Input must be a column vector");
        assert_eq!(self.rows, vector.rows, "Number of rows must match");

        let mut result_data = self.data.clone();
        for i in 0..self.rows {
            let value = vector[(i, 0)];
            for j in 0..self.cols {
                result_data[i * self.cols + j] = result_data[i * self.cols + j] + value;
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

    /// Collapses the matrix into a single column vector by summing each column.
    pub fn collapse_columns(&self) -> Matrix<T> where T: Default {
        let mut result = Matrix::from_value(self.rows, 1, T::default());
        for col in 0..self.cols {
            for row in 0..self.rows {
                result[(row, 0)] = result[(row, 0)] + self[(row, col)];
            }
        }
        result
    }
}

impl<T> Add for Matrix<T> where T: Add<Output = T> + Copy {
    type Output = Matrix<T>;

    /// Adds two matrices element-wise and returns a new matrix.
    ///
    /// # Panics
    /// Panics if the dimensions of the matrices do not match.
    fn add(self, other: Matrix<T>) -> Matrix<T> {
        self.element_wise_op(&other, |a, b| *a + *b)
    }
}

impl<T> Matrix<T> where T: Sub<Output = T> + Copy {
    /// Subtracts two matrices element-wise and returns a new matrix.
    ///
    /// # Panics
    /// Panics if the dimensions of the matrices do not match.
    pub fn sub(&self, other: &Matrix<T>) -> Matrix<T> {
        self.element_wise_op(other, |a, b| *a - *b)
    }
}

impl<T> Sub for Matrix<T> where T: Sub<Output = T> + Copy {
    type Output = Matrix<T>;

    /// Subtracts two matrices element-wise and returns a new matrix.
    ///
    /// # Panics
    /// Panics if the dimensions of the matrices do not match.
    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        self.element_wise_op(&other, |a, b| *a - *b)
    }
}

impl<T> Neg for Matrix<T> where T: Neg<Output = T> + Copy {
    type Output = Matrix<T>;

    /// Negates all elements of the matrix and returns a new matrix.
    fn neg(self) -> Matrix<T> {
        self.element_wise_map(|x| -*x)
    }
}

impl<T> Matrix<T> where T: Mul<Output = T> + Add<Output = T> + Copy + Default {
    /// Multiplies two matrices and returns a new matrix.
    ///
    /// # Panics
    /// Panics if the number of columns in the first matrix does not match the number of rows in the second matrix.
    pub fn mul(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, other.rows, "Matrix dimensions do not match for multiplication");

        let mut result_data = vec![T::default(); self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result_data[i * other.cols + j] = sum;
            }
        }
        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        }
    }
}

impl<T> Matrix<T> where T: Mul<Output = T> + Copy {
    /// Multiplies two matrices and returns a new matrix.
    ///
    /// # Panics
    /// Panics if the number of columns in the first matrix does not match the number of rows in the second matrix.
    pub fn element_wise_multiply(&self, other: &Matrix<T>) -> Matrix<T> {
        self.element_wise_op(other, |a, b| *a * *b)
    }

    /// Multiplies each element of the matrix by a scalar value.
    pub fn mul_scalar(&self, scalar: T) -> Matrix<T> {
        self.element_wise_map(|x| *x * scalar)
    }
}

impl<T> Display for Matrix<T> where T: Display {
    /// Formats the matrix with aligned columns
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Calculate column widths
        let mut col_widths = vec![0; self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = format!("{}", self[(row, col)]);
                col_widths[col] = col_widths[col].max(cell.len());
            }
        }

        // Print the matrix with aligned columns
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = format!("{}", self[(row, col)]);
                write!(f, "{:>width$} ", cell, width = col_widths[col])?;
            }
            if row < self.rows - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
