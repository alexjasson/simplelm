#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Model.h"
#include "utility.h"

/*
 * We're using a minimal gated unit (MGU) with an embedding layer and
 * letting the embedding layer size be the same as the hidden layer size.
 * Please see: https://en.wikipedia.org/wiki/Gated_recurrent_unit and
 * https://arxiv.org/pdf/1603.09420
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
    Entry *entries; // Block of memory for all matrix entries in parameters
    EmbeddingLayer e;
    HiddenLayer *h;
    OutputLayer o;
} Parameters;

typedef struct {
    Entry *entries; // Block of memory for all matrix entries in variables
    Parameters dp;  // Gradients for each parameter (dLoss/dtheta)
    Matrix *x;      // Input vector [H x 1] for each layer
    Matrix *f;      // Forget vector [H x 1] for each layer
    Matrix *h_hat;  // Candidate activation vector [H x 1] for each layer
    Matrix *z_h;    // Pre-activation vector for h_hat [H x 1] for each layer
    Matrix *h_prev; // Previous hidden state [H x 1] for each layer
    Matrix *h;      // Hidden state vectors [H x 1] for each layer
    Matrix tempV;   // Holds 3 temporary vector values [H x 3]
    Matrix tempM;   // Holds temporary matrix values [max(V,H) x H]
    Matrix dh;      // Holds upstream gradient dL/dh [H x 1] during backward pass
} Variables;

struct model
{
    size_t V, E;    // Vocabulary size (V) and embedding size (E)
    size_t H, N;    // Size of the hidden GRU layers (H) and the number of layers (N)
    Parameters p;   // Parameters (theta)
    Variables v;    // Allocated memory for calculations
};

static void softmax(Matrix output, float temperature);
static Entry sigmoid(Entry x);
static Entry dsigmoid(Entry x);
static Entry rationalTanh(Entry x);
static Entry drationalTanh(Entry x);

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
    m->p.e.W = MatrixView(V, E, addr); addr += V * E;
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->p.h[i];
        hl->W_f = MatrixView(H, H, addr); addr += H * H;
        hl->W_h = MatrixView(H, H, addr); addr += H * H;
        hl->U_f = MatrixView(H, H, addr); addr += H * H;
        hl->U_h = MatrixView(H, H, addr); addr += H * H;
        hl->b_f = MatrixView(H, 1, addr); addr += H;
        hl->b_h = MatrixView(H, 1, addr); addr += H;
    }
    m->p.o.W = MatrixView(V, H, addr); addr += V * H;
    m->p.o.b = MatrixView(V, 1, addr);

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
    m->v.dp.h = calloc(N, sizeof(HiddenLayer));
    m->v.dp.entries = calloc(ModelParameters(m), sizeof(Entry));
    if ((!m->v.dp.entries) || (!m->v.dp.h)) goto error;

    // Assign memory addresses to gradients
    addr = m->v.dp.entries;

    m->v.dp.e.W = MatrixView(V, E, addr); addr += V * E;
    for (size_t i = 0; i < N; i++)
    {
        HiddenLayer *hl = &m->v.dp.h[i];
        hl->W_f = MatrixView(H, H, addr); addr += H * H;
        hl->W_h = MatrixView(H, H, addr); addr += H * H;
        hl->U_f = MatrixView(H, H, addr); addr += H * H;
        hl->U_h = MatrixView(H, H, addr); addr += H * H;
        hl->b_f = MatrixView(H, 1, addr); addr += H;
        hl->b_h = MatrixView(H, 1, addr); addr += H;
    }
    m->v.dp.o.W = MatrixView(V, H, addr); addr += V * H;
    m->v.dp.o.b = MatrixView(V, 1, addr);

    // Allocate variables and assign memory addresses
    m->v.x      = calloc(N, sizeof(Matrix));
    m->v.f      = calloc(N, sizeof(Matrix));
    m->v.h_hat  = calloc(N, sizeof(Matrix));
    m->v.z_h    = calloc(N, sizeof(Matrix));
    m->v.h_prev = calloc(N, sizeof(Matrix));
    m->v.h      = calloc(N, sizeof(Matrix));
    size_t S = V > H ? V : H;
    m->v.entries = calloc(6 * N * H + S * H + 4 * H, sizeof(Entry));
    if (!m->v.entries || !m->v.f || !m->v.h_hat || !m->v.h_prev
        || !m->v.x || !m->v.z_h || !m->v.h) goto error;
    addr = m->v.entries;
    for (size_t i = 0; i < N; i++) {
        m->v.x[i]      = MatrixView(H, 1, addr); addr += H;
        m->v.f[i]      = MatrixView(H, 1, addr); addr += H;
        m->v.h_hat[i]  = MatrixView(H, 1, addr); addr += H;
        m->v.z_h[i]    = MatrixView(H, 1, addr); addr += H;
        m->v.h_prev[i] = MatrixView(H, 1, addr); addr += H;
        m->v.h[i]      = MatrixView(H, 1, addr); addr += H;
    }
    m->v.tempV = MatrixView(H, 3, addr); addr += 3 * H;
    m->v.tempM = MatrixView(S, H, addr); addr += S * H;
    m->v.dh = MatrixView(H, 1, addr);

    return m;

error:
    ModelFree(m);
    fprintf(stderr, "Insufficient memory!\n");
    exit(EXIT_FAILURE);
}

void ModelFree(Model m)
{
    if (!m) return;
    free(m->v.entries);
    free(m->v.x);
    free(m->v.f);
    free(m->v.h_hat);
    free(m->v.z_h);
    free(m->v.h_prev);
    free(m->v.h);
    free(m->p.h);
    free(m->p.entries);
    free(m->v.dp.h);
    free(m->v.dp.entries);
    free(m);
}

size_t ModelParameters(Model m)
{
    size_t V = m->V, E = m->E, H = m->H, N = m->N;
    return V*E + N*(4*H*H + 2*H) + V*H + V;
}

Model ModelRead(char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        return ModelNew(0, 0);
    }

    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    long minSize = 2 * sizeof(size_t) + VOCABULARY_SIZE * sizeof(Entry);
    if (fileSize < minSize) {
        fclose(f);
        return ModelNew(0, 0);
    }
    fseek(f, 0, SEEK_SET);

    size_t H, N;
    fread(&H, sizeof(size_t), 1, f);
    fread(&N, sizeof(size_t), 1, f);
    Model m = ModelNew(H, N);
    fread(m->p.entries, sizeof(Entry), ModelParameters(m), f);
    fclose(f);
    return m;
}

void ModelWrite(Model m, char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        return;
    }

    fwrite(&m->H, sizeof(size_t), 1, f);
    fwrite(&m->N, sizeof(size_t), 1, f);
    fwrite(m->p.entries, sizeof(Entry), ModelParameters(m), f);
    fclose(f);
}

void ModelReset(Model m)
{
    for (size_t l = 0; l < m->N; l++)
        MatrixZero(m->v.h[l]);
}

// Accumulates the folling variables for backward pass: {x, f, h_hat, z_h, h_prev}
void ModelForward(Model m, Token input, Matrix output)
{
    Variables *v = &m->v;
    Matrix tempH = MatrixView(m->H, 1, v->tempV.entries);

    // Embedding layer: x = W_e[input, :]^T
    Matrix x = MatrixTranspose(MatrixRow(m->p.e.W, (size_t)input));

    // Iterate over MGU layers
    for (size_t l = 0; l < m->N; l++)
    {
        HiddenLayer p = m->p.h[l]; // Hidden layer parameters
        Matrix h_prev = m->v.h[l]; // Previous hidden state

        MatrixCopy(v->x[l], x);
        MatrixCopy(v->h_prev[l], h_prev);

        // f = sigmoid(W_f * x + U_f * h_prev + b_f)
        MatrixMultiply(tempH, p.U_f, h_prev);
        MatrixMultiply(v->f[l], p.W_f, x);
        MatrixAdd(v->f[l], v->f[l], tempH);
        MatrixAdd(v->f[l], v->f[l], p.b_f);
        MatrixApply(v->f[l], sigmoid);

        // h_hat = tanh(W_h * x + U_h * (f ⊙ h_prev) + b_h)
        MatrixHadamard(tempH, v->f[l], h_prev);
        MatrixMultiply(tempH, p.U_h, tempH);
        MatrixMultiply(v->z_h[l], p.W_h, x);
        MatrixAdd(v->z_h[l], v->z_h[l], tempH);
        MatrixAdd(v->z_h[l], v->z_h[l], p.b_h);
        MatrixCopy(v->h_hat[l], v->z_h[l]);
        MatrixApply(v->h_hat[l], rationalTanh);

        // h_new = (1 - f) ⊙ h_prev + f ⊙ h_hat
        //       = h_prev - (f ⊙ h_prev) + (f ⊙ h_hat)
        MatrixHadamard(tempH, v->f[l], h_prev);
        MatrixSubtract(h_prev, h_prev, tempH);
        MatrixHadamard(tempH, v->f[l], v->h_hat[l]);
        MatrixAdd(h_prev, h_prev, tempH);

        // Next layer input is the previous layer output
        x = h_prev;
    }

    // Output layer: output = W_o * x + b_o
    MatrixMultiply(output, m->p.o.W, x);
    MatrixAdd(output, output, m->p.o.b);
}

// Accumulates gradients (dL/dtheta)
float ModelBackward(Model m, Token input, Token target, Matrix output)
{
    Variables *v = &m->v;
    size_t V = m->V, H = m->H, N = m->N;
    Matrix dh = v->dh;                                           // Upstream gradients [H x 1]
    Matrix tempH1  = MatrixView(H, 1, v->tempV.entries);         // Temporary vector [H x 1]
    Matrix tempH2  = MatrixView(H, 1, v->tempV.entries + H);     // Temporary vector [H x 1]
    Matrix tempH3  = MatrixView(H, 1, v->tempV.entries + 2 * H); // Temporary vector [H x 1]
    Matrix tempVxH = MatrixView(V, H, v->tempM.entries);         // Temporary matrix [V x H]
    Matrix tempHxH = MatrixView(H, H, v->tempM.entries);         // Temporary matrix [H x H]

    // Let y = output
    softmax(output, 1.0f);
    
    // Loss = -log(softmax(y)[target])
    float loss = -logf(MatrixGet(output, target, 0));

    // dL/dy = softmax(y) - one_hot
    MatrixSet(output, target, 0, MatrixGet(output, target, 0) - 1.0f);

    // Output layer
    // dL/dW_o += dL/dy * h^T
    MatrixMultiply(tempVxH, output, MatrixTranspose(m->v.h[N - 1]));
    MatrixAdd(m->v.dp.o.W, m->v.dp.o.W, tempVxH);

    // dL/db_o += dL/dy
    MatrixAdd(m->v.dp.o.b, m->v.dp.o.b, output);

    // dL/dh = W_o^T * dL/dy
    //       = ((dL/dy)^T * W_o)^T 
    MatrixMultiply(MatrixTranspose(dh), MatrixTranspose(output), m->p.o.W);

    // Iterate over MGU layers backwards
    for (size_t l = N; l-- > 0; )
    {
        HiddenLayer p = m->p.h[l];     // Hidden layer parameters
        HiddenLayer dp = m->v.dp.h[l]; // Hidden layer gradients

        // dL/dz_h = (dL/dh ⊙ f) ⊙ rationalTanh'(z_h)
        MatrixCopy(tempH3, v->z_h[l]);
        MatrixApply(tempH3, drationalTanh);
        MatrixHadamard(tempH1, dh, v->f[l]);
        MatrixHadamard(tempH3, tempH1, tempH3);

        // dL/dW_h += dL/dz_h * x^T
        MatrixMultiply(tempHxH, tempH3, MatrixTranspose(v->x[l]));
        MatrixAdd(dp.W_h, dp.W_h, tempHxH);

        // dL/dU_h += dL/dz_h * (f ⊙ h_prev)^T
        MatrixHadamard(tempH1, v->f[l], v->h_prev[l]);
        MatrixMultiply(tempHxH, tempH3, MatrixTranspose(tempH1));
        MatrixAdd(dp.U_h, dp.U_h, tempHxH);

        // dL/db_h += dL/dz_h
        MatrixAdd(dp.b_h, dp.b_h, tempH3);

        // dL/df = dL/dh ⊙ (h_hat - h_prev) + (U_h^T * dL/dz_h) ⊙ h_prev
        //         dL/dh ⊙ (h_hat - h_prev) + ((dL/dz_h)^T * U_h)^T ⊙ h_prev
        MatrixSubtract(tempH1, v->h_hat[l], v->h_prev[l]);
        MatrixHadamard(tempH2, dh, tempH1);
        MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(tempH3), p.U_h);
        MatrixHadamard(tempH1, tempH1, v->h_prev[l]);
        MatrixAdd(tempH2, tempH2, tempH1);

        // dL/dz_f = dL/df ⊙ sigmoid'(f)
        MatrixCopy(tempH1, v->f[l]);
        MatrixApply(tempH1, dsigmoid);
        MatrixHadamard(tempH2, tempH2, tempH1);

        // dL/dW_f += dL/dz_f * x^T
        MatrixMultiply(tempHxH, tempH2, MatrixTranspose(v->x[l]));
        MatrixAdd(dp.W_f, dp.W_f, tempHxH);

        // dL/dU_f += dL/dz_f * h_prev^T
        MatrixMultiply(tempHxH, tempH2, MatrixTranspose(v->h_prev[l]));
        MatrixAdd(dp.U_f, dp.U_f, tempHxH);

        // dL/db_f += dL/dz_f
        MatrixAdd(dp.b_f, dp.b_f, tempH2);

        // dL/dx = W_h^T * dL/dz_h + W_f^T * dL/dz_f
        //       = ((dL/dz_h)^T * W_h)^T + ((dL/dz_f)^T * W_f)^T
        MatrixMultiply(MatrixTranspose(dh), MatrixTranspose(tempH3), p.W_h);
        MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(tempH2), p.W_f);
        MatrixAdd(dh, dh, tempH1);

        // dL/dx becomes dL/dh for the previous layer
    }

    // Embedding layer
    // dL/dW_e[input, :] += dL/dx
    Matrix dW_e = MatrixTranspose(MatrixRow(m->v.dp.e.W, (size_t)input));
    MatrixAdd(dW_e, dW_e, dh);

    return loss;
}

// Softmax sampling via inverse CDF
Token ModelSample(Model m, Matrix output, float temperature)
{
    size_t V = m->V;

    // Return most likely token if temperature <= 0
    if (temperature <= 0.0f) {
        size_t argmax = 0;
        for (size_t i = 1; i < V; i++)
            if (MatrixGet(output, i, 0) > MatrixGet(output, argmax, 0))
                argmax = i;
        return (Token)argmax;
    }

    softmax(output, temperature);

    float r = randomFloat(0.0f, 1.0f);
    Entry cdf = 0.0f;
    for (size_t i = 0; i < V; i++) {
        cdf += MatrixGet(output, i, 0);
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

// Sigmoid derivative
static Entry dsigmoid(Entry x)
{
    return x * (1.0f - x);
}

// Rational tanh approximation, valid for x in [-3, 3]
static Entry rationalTanh(Entry x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    Entry x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Rational tanh derivative, f(x) = x(27 + x^2)/(27 + 9x^2)
//                           f'(x) = (x^2 - 9)^2 / 9(x^2 + 3)^2
static Entry drationalTanh(Entry x) {
    if (x <= -3.0f || x >= 3.0f) return 0.0f;
    Entry x2 = x * x;
    Entry numerator = x2 - 9.0f;
    Entry denominator = x2 + 3.0f;
    return (numerator * numerator) / (9.0f * denominator * denominator);
}

// Softmax with temperature
static void softmax(Matrix output, float temperature)
{
    size_t n = output.rows;

    Entry max = MatrixGet(output, 0, 0);
    for (size_t i = 1; i < n; i++) {
        Entry e = MatrixGet(output, i, 0);
        if (e > max) max = e;
    }

    Entry sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        Entry e = expf((MatrixGet(output, i, 0) - max) / temperature);
        MatrixSet(output, i, 0, e);
        sum += e;
    }
    for (size_t i = 0; i < n; i++)
        MatrixSet(output, i, 0, MatrixGet(output, i, 0) / sum);
}
