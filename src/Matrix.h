#ifndef MATRIX_H
#define MATRIX_H

typedef float Entry;
typedef struct {
    size_t rows;
    size_t cols;
    Entry *entries;
} Matrix;
typedef float (*MatrixFunction)(Entry);

/*
 * Creates a matrix, requires already allocated memory for entries
 */
Matrix matrixNew(size_t rows, size_t cols, Entry *entries);

/*
 * out = A + B, assumes all parameters given are correct
 */
void matrixAdd(Matrix out, Matrix A, Matrix B);

/*
 * out = A - B, assumes all parameters given are correct
 */
void matrixSubtract(Matrix out, Matrix A, Matrix B);

/*
 * out = A ⊙ B, assumes all parameters given are correct
 */
void matrixHadamard(Matrix out, Matrix A, Matrix B);

/*
 * out = A * B, assumes all parameters given are correct
 */
void matrixMultiply(Matrix out, Matrix A, Matrix B);

/*
 * Xavier initialize a matrix
 */
void matrixXavier(Matrix A);

/*
 * Applies a single variable function to each element in the matrix
 */
void matrixApply(Matrix A, MatrixFunction f);

/*
 * Sets all elements in the matrix to zero
 */
void matrixZero(Matrix A);

/* Returns the transpose of the given row in matrix A, note that it
 * shares memory with matrix A. Use with caution.
 */
Matrix matrixView(Matrix A, size_t row);

#endif
