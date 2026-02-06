#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Model.h"

/*
 * We're using a minimal gated unit (MGU) with an embedding layer and
 * letting the embedding layer size be the same as the hidden layer size.
 * Please see: https://en.wikipedia.org/wiki/Gated_recurrent_unit
 */

typedef struct {
    Matrix W; // [V x E], Note E = H
} EmbeddingLayer;

typedef struct {
    Matrix W_f, W_h; // [H x H]
    Matrix U_f, U_h; // [H x H]
    Matrix b_f, b_h; // [H x 1]
} HiddenLayer;

typedef struct {
    Matrix W; // [V x H]
    Matrix b; // [V x 1]
} OutputLayer;

typedef struct {
    Entry *entries;   // Block of memory for all matrix entries in parameters
    EmbeddingLayer e;
    HiddenLayer *h;
    OutputLayer o;
} Parameters;

typedef Parameters Gradients; // Same data structure for our purposes

typedef struct {
    Entry *entries; // Block of memory for all matrix entries
    Matrix f;       // Forget vector [H x 1]
    Matrix h_hat;   // Candidate activation vector [H x 1]
    Matrix temp;    // Holds temporary values [H x 1]
} Variables;

typedef struct {
    Entry *entries; // Block of memory for all matrix entries
    Matrix *h;      // Hidden state vectors, [H x 1] for each layer
} HiddenState;

struct model
{
    size_t V, E;    // Vocabulary size (V) and embedding size (E)
    size_t H, N;    // Size of the hidden GRU layers (H) and the number of layers (N)
    Parameters p;   // Parameters (theta)
    Gradients g;    // Gradients (dLoss/dtheta)
    HiddenState hs; // Hidden state
    Variables v;    // Allocated memory for calculations
};

// Matrix API functions and helpers
static void matrixApply(Matrix A, MatrixFunction f);
static void matrixAdd(Matrix out, Matrix A, Matrix B);
static void matrixSubtract(Matrix out, Matrix A, Matrix B);
static void matrixHadamard(Matrix out, Matrix A, Matrix B);
static void matrixMultiply(Matrix out, Matrix A, Matrix B);
static Matrix matrixNew(size_t rows, size_t cols, Entry *entries);
static void matrixXavier(Matrix A);
static void matrixZero(Matrix A);
static Matrix matrixView(Matrix A, size_t row);
static uint32_t xorshift();
static Entry randomEntry(Entry bound);

// Model API helpers
static Entry sigmoid(Entry x);
static Entry rationalTanh(Entry x);

Model ModelNew(int hiddenSize, int numLayers)
{
    Model m = calloc(1, sizeof(struct model));
    if (!m) return NULL;

    m->V = VOCABULARY_SIZE;
    m->E = hiddenSize; // E = H for simplicity
    m->H = hiddenSize;
    m->N = numLayers;
    size_t V = m->V, E = m->E, H = m->H, N = m->N;
    Entry *addr;

    // Allocate parameters
    m->p.h = calloc(N, sizeof(HiddenLayer));
    m->p.entries = calloc(ModelParameters(m), sizeof(Entry));
    if ((!m->p.entries) || (!m->p.h)) goto error;

    // Assign memory addresses to parameters
    addr = m->p.entries;
    m->p.e.W = matrixNew(V, E, addr); addr += V * E;
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->p.h[i];
        hl->W_f = matrixNew(H, H, addr); addr += H * H;
        hl->W_h = matrixNew(H, H, addr); addr += H * H;
        hl->U_f = matrixNew(H, H, addr); addr += H * H;
        hl->U_h = matrixNew(H, H, addr); addr += H * H;
        hl->b_f = matrixNew(H, 1, addr); addr += H;
        hl->b_h = matrixNew(H, 1, addr); addr += H;
    }
    m->p.o.W = matrixNew(V, H, addr); addr += V * H;
    m->p.o.b = matrixNew(V, 1, addr);

    // Initialize parameters - Xavier for weight matrices, leave biases as 0
    matrixXavier(m->p.e.W);
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->p.h[i];
        matrixXavier(hl->W_f);
        matrixXavier(hl->W_h);
        matrixXavier(hl->U_f);
        matrixXavier(hl->U_h);
    }
    matrixXavier(m->p.o.W);

    // Allocate gradients
    m->g.h = calloc(N, sizeof(HiddenLayer));
    m->g.entries = calloc(ModelParameters(m), sizeof(Entry));
    if ((!m->g.entries) || (!m->g.h)) goto error;

    // Assign memory addresses to gradients
    addr = m->g.entries;

    m->g.e.W = matrixNew(V, E, addr); addr += V * E;
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->g.h[i];
        hl->W_f = matrixNew(H, H, addr); addr += H * H;
        hl->W_h = matrixNew(H, H, addr); addr += H * H;
        hl->U_f = matrixNew(H, H, addr); addr += H * H;
        hl->U_h = matrixNew(H, H, addr); addr += H * H;
        hl->b_f = matrixNew(H, 1, addr); addr += H;
        hl->b_h = matrixNew(H, 1, addr); addr += H;
    }
    m->g.o.W = matrixNew(V, H, addr); addr += V * H;
    m->g.o.b = matrixNew(V, 1, addr);

    // Allocate hidden states and assign memory addresses
    m->hs.h = calloc(N, sizeof(Matrix));
    m->hs.entries = calloc(N * H, sizeof(Entry));
    if ((!m->hs.entries) || (!m->hs.h)) goto error;
    for (size_t i = 0; i < N; i++)
        m->hs.h[i] = matrixNew(H, 1, m->hs.entries + i * H);

    // Allocate variables and assign memory addresses
    m->v.entries = calloc(3 * H, sizeof(Entry));
    if (!m->v.entries) goto error;
    addr = m->v.entries;
    m->v.f     = matrixNew(H, 1, addr); addr += H;
    m->v.h_hat = matrixNew(H, 1, addr); addr += H;
    m->v.temp  = matrixNew(H, 1, addr);

    return m;

error:
    ModelFree(m);
    fprintf(stderr, "Insufficient memory!\n");
    exit(EXIT_FAILURE);
}

void ModelFree(Model m)
{
    if (!m) return;
    free(m->hs.entries);
    free(m->hs.h);
    free(m->v.entries);
    free(m->p.h);
    free(m->p.entries);
    free(m->g.h);
    free(m->g.entries);
    free(m);
}

size_t ModelParameters(Model m)
{
    size_t V = m->V, E = m->E, H = m->H, N = m->N;
    return V*E + N*(4*H*H + 2*H) + V*H + V;
}

Model ModelRead(char *path)
{
    // TODO
    return ModelNew(0, 0);
}

void ModelWrite(Model m, char *path)
{
    // TODO
    return;
}

void ModelReset(Model m)
{
    for (size_t l = 0; l < m->N; l++)
        matrixZero(m->hs.h[l]);
}

void ModelForward(Model m, Token input, Matrix output)
{
    Variables *v = &m->v;

    // Embedding layer: x = W_e[input, :]^T
    Matrix x = matrixView(m->p.e.W, (size_t)input);

    // Iterate over MGU layers
    for (size_t l = 0; l < m->N; l++)
    {
        HiddenLayer hl = m->p.h[l];
        Matrix h_prev  = m->hs.h[l];

        // f = sigmoid(W_f * x + U_f * h_prev + b_f)
        matrixMultiply(v->temp, hl.U_f, h_prev);
        matrixMultiply(v->f, hl.W_f, x);
        matrixAdd(v->f, v->f, v->temp);
        matrixAdd(v->f, v->f, hl.b_f);
        matrixApply(v->f, sigmoid);

        // h_hat = tanh(W_h * x + U_h * (f ⊙ h_prev) + b_h)
        matrixHadamard(v->temp, v->f, h_prev);
        matrixMultiply(v->temp, hl.U_h, v->temp);
        matrixMultiply(v->h_hat, hl.W_h, x);
        matrixAdd(v->h_hat, v->h_hat, v->temp);
        matrixAdd(v->h_hat, v->h_hat, hl.b_h);
        matrixApply(v->h_hat, rationalTanh);

        // h_new = (1 - f) ⊙ h_prev + f ⊙ h_hat
        //       = h_prev - (f ⊙ h_prev) + (f ⊙ h_hat)
        matrixHadamard(v->temp, v->f, h_prev);
        matrixSubtract(h_prev, h_prev, v->temp);
        matrixHadamard(v->temp, v->f, v->h_hat);
        matrixAdd(h_prev, h_prev, v->temp);

        // Next layer input is the previous layer output
        x = h_prev;
    }

    // Output layer: output = W_o * x + b_o
    matrixMultiply(output, m->p.o.W, x);
    matrixAdd(output, output, m->p.o.b);
}

Token ModelSample(Model m, Matrix output)
{
    return 0;
}

// Sigmoid function
static inline Entry sigmoid(Entry x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Rational tanh approximation, valid for x in [-3, 3]
static Entry rationalTanh(Entry x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    Entry x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Xorshift algorithm
static uint32_t xorshift()
{
  static uint32_t state = 0;
  static int seeded = 0;

  // Seed only once
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
static inline Entry randomEntry(Entry bound)
{
    return ((xorshift() >> 8) * (2.0f / 16777216.0f) - 1.0f) * bound;
}

// Xavier initialize a matrix
static void matrixXavier(Matrix A) {
    size_t size = A.rows * A.cols;
    Entry bound = sqrtf(6.0f / (A.rows + A.cols));
    for (size_t i = 0; i < size; i++) {
        A.entries[i] = randomEntry(bound);
    }
}

// // Creates a matrix from existing data
static Matrix matrixNew(size_t rows, size_t cols, Entry *entries)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.entries = entries;
    return m;
}

// out = A * B, assumes all parameters given are correct
static void matrixMultiply(Matrix out, Matrix A, Matrix B)
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

// out = A + B, assumes all parameters given are correct
static void matrixAdd(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] + B.entries[i];
}

// out = A - B, assumes all parameters given are correct
static void matrixSubtract(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] - B.entries[i];
}

// out = A ⊙ B, assumes all parameters given are correct
static void matrixHadamard(Matrix out, Matrix A, Matrix B)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        out.entries[i] = A.entries[i] * B.entries[i];
}

// Applies a specific single variable function to each element in the matrix
static void matrixApply(Matrix A, MatrixFunction f)
{
    size_t n = A.rows * A.cols;
    for (size_t i = 0; i < n; i++)
        A.entries[i] = f(A.entries[i]);
}

// Sets all elements in the matrix to zero
static void matrixZero(Matrix A)
{
    memset(A.entries, 0, A.rows * A.cols * sizeof(Entry));
}

// Returns the transpose of the given row in matrix A, note that it
// shares memory with matrix A. Use with caution.
static inline Matrix matrixView(Matrix A, size_t row)
{
    Matrix m;
    m.rows = A.cols;
    m.cols = 1;
    m.entries = A.entries + row * A.cols;
    return m;
}

