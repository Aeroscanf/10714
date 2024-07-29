#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void matrix_mutltiply(size_t row1, size_t col1, size_t row2, size_t col2, float *A, float *B, float *C){
  if (col1 != row2){
    printf("Error: Incompatible matrix dimensions for multiplication\n");
    return;
  }
  for(size_t i = 0; i < row1; i++){
    for(size_t j = 0; j < col2; j++){
      C[i*col2 + j] = 0;
      for(size_t k = 0; k < col1; k++){
        C[i*col2 + j] += A[i*col1 + k] * B[k*col2 + j];
      }
    }
  }
}

void matrix_softmax_prob(size_t row, size_t col, float *A){
  for(size_t i = 0; i < row; i++){
    float max_val = *std::max_element(A, A + col);
    float sum = 0;
    for(size_t j = 0; j < col; j++){
      A[i*col + j] = exp(A[i*col + j] - max_val);
      sum += A[i*col + j];
    }
    for(size_t j = 0; j < col; j++){
      A[i*col + j] = A[i*col + j] / sum;
    }
  }
}

void to_one_hot(size_t row, size_t col, float *A, const unsigned char *y){
  for(size_t i = 0; i < row; i++){
    size_t i_th = static_cast<size_t>(y[i]);
    for(size_t j = 0; j < col; j++){
      if(j == i_th){
        A[i*col + j] = 1;
        break;
      }
    }
  }
}

void substract(size_t row, size_t col, float *A, float *B){
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      A[i*col + j] -= B[i*col + j];
    }
  }
}

void transpose(size_t row, size_t col, float *A, float *B){
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      B[j*row + i] = A[i*col + j];
    }
  }
}

void num_matrix_multiply(size_t row, size_t col, float lr, float *A){
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      A[col*i + j] *= lr;
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): posize_ter to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): posize_ter to y data, of size m
     *     theta (float *): posize_ter to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (size_t): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t i = 0; i < m; i += batch){
      size_t current_batch = std::min(batch, m - i);
      const float *X_batch = X + i;
      const unsigned char *y_batch = y + i;

      //compute softmax_prob
      double *prob = new double[current_batch * k];
      matrix_mutltiply(current_batch, n, n, k, const_cast<double*>(X_batch), theta, prob);
      matrix_softmax_prob(current_batch, k, prob);

      //compute y_one_hot
      double *y_one_hot =new double[current_batch*k];
      memset(y_one_hot, 0, sizeof(current_batch*k));
      to_one_hot(current_batch, k, y_one_hot, const_cast<unsigned char*>(y_batch));

      //compute grad
      float *X_tp = (float*)malloc(current_batch *n * sizeof(float));
      transpose(current_batch, n, const_cast<float*>(X_batch), X_tp);
      substract(current_batch, k, prob, y_one_hot);
      delete[] y_one_hot;
      float *grad = (float*)malloc(n*k*sizeof(float)); 
      matrix_mutltiply(n, current_batch, current_batch, k, X_tp, prob, grad);
      float corr = 1 / float(current_batch);
      num_matrix_multiply(n, k, corr, grad);
      free(X_tp);

      num_matrix_multiply(n, k, lr, grad);
      substract(n, k, theta, grad);
      free(grad);
    }
    /// END YOUR CODE
}


//code of cluad3.5
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    // Allocate memory for temporary arrays
    float* Z = new float[batch * k];
    float* gradients = new float[n * k];

    // Iterate over minibatches
    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);
        
        // Compute Z = X * theta (for current batch)
        for (size_t b = 0; b < current_batch_size; ++b) {
            for (size_t j = 0; j < k; ++j) {
                Z[b*k + j] = 0;
                for (size_t l = 0; l < n; ++l) {
                    Z[b*k + j] += X[(i+b)*n + l] * theta[l*k + j];
                }
            }
        }

        // Apply softmax to Z
        for (size_t b = 0; b < current_batch_size; ++b) {
            float max_val = Z[b*k];
            for (size_t j = 1; j < k; ++j) {
                if (Z[b*k + j] > max_val) max_val = Z[b*k + j];
            }
            float sum_exp = 0;
            for (size_t j = 0; j < k; ++j) {
                Z[b*k + j] = std::exp(Z[b*k + j] - max_val);
                sum_exp += Z[b*k + j];
            }
            for (size_t j = 0; j < k; ++j) {
                Z[b*k + j] /= sum_exp;
            }
        }

        // Compute gradients
        std::fill(gradients, gradients + n*k, 0);
        for (size_t b = 0; b < current_batch_size; ++b) {
            for (size_t j = 0; j < k; ++j) {
                float indicator = (j == y[i+b]) ? 1.0f : 0.0f;
                float diff = Z[b*k + j] - indicator;
                for (size_t l = 0; l < n; ++l) {
                    gradients[l*k + j] += X[(i+b)*n + l] * diff;
                }
            }
        }

        // Update theta
        for (size_t l = 0; l < n*k; ++l) {
            theta[l] -= lr * gradients[l] / current_batch_size;
        }
    }

    // Free allocated memory
    delete[] Z;
    delete[] gradients;
}



/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           size_t batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
