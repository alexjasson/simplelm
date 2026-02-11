#ifndef MATRIX_H
#define MATRIX_H

typedef float Entry;
typedef Entry (*EntryFunction)(Entry);
typedef struct {
    size_t rows;
    size_t cols;
    Entry *entries;
} Matrix;

/*
 * Allocates a new matrix on the heap with all entries as 0
 */
Matrix MatrixNew(size_t rows, size_t cols);

/*
 * Frees a matrix from memory
 */
void MatrixFree(Matrix A);

/*
 * Creates a matrix view over existing memory.
 */
Matrix MatrixView(size_t rows, size_t cols, Entry *entries);

/*
 * out = A + B, assumes all parameters given are correct
 */
void MatrixAdd(Matrix out, Matrix A, Matrix B);

/*
 * out = A - B, assumes all parameters given are correct
 */
void MatrixSubtract(Matrix out, Matrix A, Matrix B);

/*
 * out = A ⊙ B, assumes all parameters given are correct
 */
void MatrixHadamard(Matrix out, Matrix A, Matrix B);

/*
 * out = A * B, assumes all parameters given are correct
 */
void MatrixMultiply(Matrix out, Matrix A, Matrix B);

/*
 * Xavier initialize a matrix
 */
void MatrixXavier(Matrix A);

/*
 * Applies a single variable function to each entry in the matrix
 */
void MatrixApply(Matrix A, EntryFunction f);

/*
 * Sets all elements in the matrix to zero
 */
void MatrixZero(Matrix A);

/* Returns the transpose of the given row in matrix A, note that it
 * shares memory with matrix A. Use with caution.
 */
Matrix MatrixSlice(Matrix A, size_t row);

/*
 * Returns the entry at the given row and column.
 */
Entry MatrixGet(Matrix A, size_t row, size_t col);

/*
 * Sets the entry at the given row and column.
 */
void MatrixSet(Matrix A, size_t row, size_t col, Entry value);

#endif
