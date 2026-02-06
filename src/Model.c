#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Model.h"
#include "utility.h"

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
    m->p.e.W = MatrixNew(V, E, addr); addr += V * E;
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->p.h[i];
        hl->W_f = MatrixNew(H, H, addr); addr += H * H;
        hl->W_h = MatrixNew(H, H, addr); addr += H * H;
        hl->U_f = MatrixNew(H, H, addr); addr += H * H;
        hl->U_h = MatrixNew(H, H, addr); addr += H * H;
        hl->b_f = MatrixNew(H, 1, addr); addr += H;
        hl->b_h = MatrixNew(H, 1, addr); addr += H;
    }
    m->p.o.W = MatrixNew(V, H, addr); addr += V * H;
    m->p.o.b = MatrixNew(V, 1, addr);

    // Initialize parameters - Xavier for weight matrices, leave biases as 0
    MatrixXavier(m->p.e.W);
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->p.h[i];
        MatrixXavier(hl->W_f);
        MatrixXavier(hl->W_h);
        MatrixXavier(hl->U_f);
        MatrixXavier(hl->U_h);
    }
    MatrixXavier(m->p.o.W);

    // Allocate gradients
    m->g.h = calloc(N, sizeof(HiddenLayer));
    m->g.entries = calloc(ModelParameters(m), sizeof(Entry));
    if ((!m->g.entries) || (!m->g.h)) goto error;

    // Assign memory addresses to gradients
    addr = m->g.entries;

    m->g.e.W = MatrixNew(V, E, addr); addr += V * E;
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->g.h[i];
        hl->W_f = MatrixNew(H, H, addr); addr += H * H;
        hl->W_h = MatrixNew(H, H, addr); addr += H * H;
        hl->U_f = MatrixNew(H, H, addr); addr += H * H;
        hl->U_h = MatrixNew(H, H, addr); addr += H * H;
        hl->b_f = MatrixNew(H, 1, addr); addr += H;
        hl->b_h = MatrixNew(H, 1, addr); addr += H;
    }
    m->g.o.W = MatrixNew(V, H, addr); addr += V * H;
    m->g.o.b = MatrixNew(V, 1, addr);

    // Allocate hidden states and assign memory addresses
    m->hs.h = calloc(N, sizeof(Matrix));
    m->hs.entries = calloc(N * H, sizeof(Entry));
    if ((!m->hs.entries) || (!m->hs.h)) goto error;
    for (size_t i = 0; i < N; i++)
        m->hs.h[i] = MatrixNew(H, 1, m->hs.entries + i * H);

    // Allocate variables and assign memory addresses
    m->v.entries = calloc(3 * H, sizeof(Entry));
    if (!m->v.entries) goto error;
    addr = m->v.entries;
    m->v.f     = MatrixNew(H, 1, addr); addr += H;
    m->v.h_hat = MatrixNew(H, 1, addr); addr += H;
    m->v.temp  = MatrixNew(H, 1, addr);

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
        MatrixZero(m->hs.h[l]);
}

void ModelForward(Model m, Token input, Matrix output)
{
    Variables *v = &m->v;

    // Embedding layer: x = W_e[input, :]^T
    Matrix x = MatrixView(m->p.e.W, (size_t)input);

    // Iterate over MGU layers
    for (size_t l = 0; l < m->N; l++)
    {
        HiddenLayer hl = m->p.h[l];
        Matrix h_prev  = m->hs.h[l];

        // f = sigmoid(W_f * x + U_f * h_prev + b_f)
        MatrixMultiply(v->temp, hl.U_f, h_prev);
        MatrixMultiply(v->f, hl.W_f, x);
        MatrixAdd(v->f, v->f, v->temp);
        MatrixAdd(v->f, v->f, hl.b_f);
        MatrixApply(v->f, sigmoid);

        // h_hat = tanh(W_h * x + U_h * (f ⊙ h_prev) + b_h)
        MatrixHadamard(v->temp, v->f, h_prev);
        MatrixMultiply(v->temp, hl.U_h, v->temp);
        MatrixMultiply(v->h_hat, hl.W_h, x);
        MatrixAdd(v->h_hat, v->h_hat, v->temp);
        MatrixAdd(v->h_hat, v->h_hat, hl.b_h);
        MatrixApply(v->h_hat, rationalTanh);

        // h_new = (1 - f) ⊙ h_prev + f ⊙ h_hat
        //       = h_prev - (f ⊙ h_prev) + (f ⊙ h_hat)
        MatrixHadamard(v->temp, v->f, h_prev);
        MatrixSubtract(h_prev, h_prev, v->temp);
        MatrixHadamard(v->temp, v->f, v->h_hat);
        MatrixAdd(h_prev, h_prev, v->temp);

        // Next layer input is the previous layer output
        x = h_prev;
    }

    // Output layer: output = W_o * x + b_o
    MatrixMultiply(output, m->p.o.W, x);
    MatrixAdd(output, output, m->p.o.b);
}

// Softmax sampling with temperature via inverse CDF
Token ModelSample(Model m, Matrix output, float temperature)
{
    size_t V = m->V;

    size_t argmax = 0;
    Entry max = MatrixGet(output, 0, 0);
    for (size_t i = 1; i < V; i++) {
        Entry e = MatrixGet(output, i, 0);
        if (e > max) {
            max = e;
            argmax = i;
        }
    }

    if (!(temperature > 0.0f))
        return (Token)argmax;

    Entry sum = 0.0f;
    for (size_t i = 0; i < V; i++) {
        Entry e = MatrixGet(output, i, 0);
        sum += expf((e - max) / temperature);
    }
    if (!(sum > 0.0f))
        return (Token)argmax;

    float r = randomFloat(0.0f, (float)sum);

    Entry cdf = 0.0f;
    for (size_t i = 0; i < V; i++) {
        Entry e = MatrixGet(output, i, 0);
        cdf += expf((e - max) / temperature);
        if (r < cdf)
            return (Token)i;
    }

    return (Token)(V - 1);
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
