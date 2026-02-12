#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "Matrix.h"
#include "ThreadPool.h"
#include "utility.h"

typedef struct {
    Matrix out, A, B;
} MatrixArguments;

typedef struct {
    Matrix A;
    EntryFunction f;
} EnryArguments;

// Map functions that operate on a subset of entries [start, end)
static void applySubset(void *arg, size_t start, size_t end);
static void addSubset(void *arg, size_t start, size_t end);
static void subtractSubset(void *arg, size_t start, size_t end);
static void hadamardSubset(void *arg, size_t start, size_t end);
static void multiplySubset(void *arg, size_t start, size_t end);

// Singleton pattern to initialize thread pool once then free on exit.
// Number of threads set to number of cores in CPU.
static ThreadPool pool;
static ThreadPool getPool(void);
static void freePool(void);

Matrix MatrixNew(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.entries = calloc(rows * cols, sizeof(Entry));
    if (!m.entries) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    return m;
}

void MatrixFree(Matrix A)
{
    free(A.entries);
}

Matrix MatrixView(size_t rows, size_t cols, Entry *entries)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.entries = entries;
    return m;
}

void MatrixApply(Matrix A, EntryFunction f)
{
    EnryArguments args = {A, f};
    ThreadPoolMap(getPool(), applySubset, &args, A.rows * A.cols);
}

void MatrixAdd(Matrix out, Matrix A, Matrix B)
{
    MatrixArguments args = {out, A, B};
    ThreadPoolMap(getPool(), addSubset, &args, A.rows * A.cols);
}

void MatrixSubtract(Matrix out, Matrix A, Matrix B)
{
    MatrixArguments args = {out, A, B};
    ThreadPoolMap(getPool(), subtractSubset, &args, A.rows * A.cols);
}

void MatrixHadamard(Matrix out, Matrix A, Matrix B)
{
    MatrixArguments args = {out, A, B};
    ThreadPoolMap(getPool(), hadamardSubset, &args, A.rows * A.cols);
}

void MatrixMultiply(Matrix out, Matrix A, Matrix B)
{
    MatrixArguments args = {out, A, B};
    ThreadPoolMap(getPool(), multiplySubset, &args, A.rows);
}

Matrix MatrixTranspose(Matrix A)
{
    return MatrixView(A.cols, A.rows, A.entries);
}

void MatrixXavier(Matrix A)
{
    size_t size = A.rows * A.cols;
    Entry bound = sqrtf(6.0f / (A.rows + A.cols));
    for (size_t i = 0; i < size; i++)
        A.entries[i] = (Entry)randomFloat(-bound, bound);
}

void MatrixCopy(Matrix out, Matrix in)
{
    memcpy(out.entries, in.entries, in.rows * in.cols * sizeof(Entry));
}

void MatrixZero(Matrix A)
{
    memset(A.entries, 0, A.rows * A.cols * sizeof(Entry));
}

Matrix MatrixRow(Matrix A, size_t row)
{
    return MatrixView(1, A.cols, A.entries + row * A.cols);
}

Entry MatrixGet(Matrix A, size_t row, size_t col)
{
    return A.entries[row * A.cols + col];
}

void MatrixSet(Matrix A, size_t row, size_t col, Entry value)
{
    A.entries[row * A.cols + col] = value;
}

static void applySubset(void *arg, size_t start, size_t end)
{
    EnryArguments *args = arg;
    for (size_t i = start; i < end; i++)
        args->A.entries[i] = args->f(args->A.entries[i]);
}

static void addSubset(void *arg, size_t start, size_t end)
{
    MatrixArguments *args = arg;
    for (size_t i = start; i < end; i++)
        args->out.entries[i] = args->A.entries[i] + args->B.entries[i];
}

static void subtractSubset(void *arg, size_t start, size_t end)
{
    MatrixArguments *args = arg;
    for (size_t i = start; i < end; i++)
        args->out.entries[i] = args->A.entries[i] - args->B.entries[i];
}

static void hadamardSubset(void *arg, size_t start, size_t end)
{
    MatrixArguments *args = arg;
    for (size_t i = start; i < end; i++)
        args->out.entries[i] = args->A.entries[i] * args->B.entries[i];
}

static void multiplySubset(void *arg, size_t start, size_t end)
{
    MatrixArguments *args = arg;
    for (size_t i = start; i < end; i++)
    {
        Entry *A_i = args->A.entries + i * args->A.cols;
        Entry *C_i = args->out.entries + i * args->B.cols;

        for (size_t k = 0; k < args->B.cols; k++)
            C_i[k] = 0.0f;

        for (size_t j = 0; j < args->A.cols; j++)
        {
            Entry a = A_i[j];
            Entry *B_j = args->B.entries + j * args->B.cols;
            for (size_t k = 0; k < args->B.cols; k++)
                C_i[k] += a * B_j[k];
        }
    }
}

static void freePool(void)
{
    ThreadPoolFree(pool);
}

static ThreadPool getPool(void)
{
    if (!pool) {
        pool = ThreadPoolNew(sysconf(_SC_NPROCESSORS_ONLN));
        atexit(freePool);
    }
    return pool;
}
