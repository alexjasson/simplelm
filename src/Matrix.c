#include <string.h>
#include <math.h>
#include "Matrix.h"
#include "utility.h"

Matrix MatrixNew(size_t rows, size_t cols, Entry *entries)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.entries = entries;
    return m;
}

void MatrixApply(Matrix A, MatrixFunction f)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        A.entries[i] = f(A.entries[i]);
}

void MatrixAdd(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] + B.entries[i];
}

void MatrixSubtract(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] - B.entries[i];
}

void MatrixHadamard(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] * B.entries[i];
}

void MatrixMultiply(Matrix out, Matrix A, Matrix B)
{
    for (size_t i = 0; i < A.rows; i++)
    {
        Entry *A_i = A.entries + i * A.cols;
        Entry *C_i = out.entries + i * B.cols;

        for (size_t k = 0; k < B.cols; k++)
            C_i[k] = 0.0f;

        for (size_t j = 0; j < A.cols; j++)
        {
            Entry a = A_i[j];
            Entry *B_j = B.entries + j * B.cols;
            for (size_t k = 0; k < B.cols; k++)
                C_i[k] += a * B_j[k];
        }
    }
}

void MatrixXavier(Matrix A)
{
    size_t size = A.rows * A.cols;
    Entry bound = sqrtf(6.0f / (A.rows + A.cols));
    for (size_t i = 0; i < size; i++)
        A.entries[i] = (Entry)randomFloat(-bound, bound);
}

void MatrixZero(Matrix A)
{
    memset(A.entries, 0, A.rows * A.cols * sizeof(Entry));
}

Matrix MatrixView(Matrix A, size_t row)
{
    Matrix m;
    m.rows = A.cols;
    m.cols = 1;
    m.entries = A.entries + row * A.cols;
    return m;
}

Entry MatrixGet(Matrix A, size_t row, size_t col)
{
    return A.entries[row * A.cols + col];
}

void MatrixSet(Matrix A, size_t row, size_t col, Entry value)
{
    A.entries[row * A.cols + col] = value;
}
