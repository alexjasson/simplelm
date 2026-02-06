#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Matrix.h"

static uint32_t xorshift();
static Entry randomEntry(Entry bound);

Matrix matrixNew(size_t rows, size_t cols, Entry *entries)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.entries = entries;
    return m;
}

void matrixApply(Matrix A, MatrixFunction f)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        A.entries[i] = f(A.entries[i]);
}

void matrixAdd(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] + B.entries[i];
}

void matrixSubtract(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] - B.entries[i];
}

void matrixHadamard(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] * B.entries[i];
}

void matrixMultiply(Matrix out, Matrix A, Matrix B)
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

void matrixXavier(Matrix A)
{
    size_t size = A.rows * A.cols;
    Entry bound = sqrtf(6.0f / (A.rows + A.cols));
    for (size_t i = 0; i < size; i++)
        A.entries[i] = randomEntry(bound);
}

void matrixZero(Matrix A)
{
    memset(A.entries, 0, A.rows * A.cols * sizeof(Entry));
}

Matrix matrixView(Matrix A, size_t row)
{
    Matrix m;
    m.rows = A.cols;
    m.cols = 1;
    m.entries = A.entries + row * A.cols;
    return m;
}

// Xorshift algorithm
static uint32_t xorshift()
{
    static uint32_t state = 0;
    static int seeded = 0;

    if (!seeded)
    {
        state = (uint32_t)time(NULL);
        seeded = 1;
    }

    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    state = x;
    return x;
}

// Returns a random entry in [-bound, +bound]
static Entry randomEntry(Entry bound)
{
    return ((xorshift() >> 8) * (2.0f / 16777216.0f) - 1.0f) * bound;
}
