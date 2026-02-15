#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Model.h"
#include "utility.h"

// Adam optimizer hyperparameters
#define BETA1    0.9f
#define BETA2    0.999f
#define EPSILON  1e-8f
#define CLIP     1.0f

/*
 * We're using a minimal gated unit (MGU) with an embedding layer and
 * letting the embedding layer size be the same as the hidden layer size.
 * Please see: https://arxiv.org/pdf/1603.09420
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
    Entry *entries;  // Block of memory for vectors
    Matrix *h;       // N persistent hidden state vectors [H x 1]
} HiddenState;

typedef struct {
    Entry *entries;  // Block of memory for all matrix entries in variables
    Parameters dp;   // Gradients for each parameter (dLoss/dtheta)
    Matrix *x;       // T*N Input vectors [H x 1]
    Matrix *f;       // T*N Forget vectors [H x 1]
    Matrix *h_hat;   // T*N Candidate activation vectors [H x 1]
    Matrix *z_h;     // T*N Pre-activation vectors for h_hat [H x 1]
    Matrix *h;       // T*N Hidden state vectors [H x 1]
    Matrix *y;       // T Output logit vectors [V x 1]
    Matrix tempV;    // Holds 3 temporary vector values [H x 3]
    Matrix tempM;    // Holds temporary matrix values [max(V,H) x H]
    Matrix *dh;      // N gradient vectors dL/dh [H x 1]
    Matrix *h_start; // N pre-forward hidden state vectors [H x 1]
    float *m;        // Adam first moment estimates
    float *v;        // Adam second moment estimates
    int t;           // Adam timestep
} Variables;

struct model
{
    size_t V, E;     // Vocabulary size (V) and embedding size (E)
    size_t H, N;     // Size of the hidden GRU layers (H) and the number of layers (N)
    size_t T;        // Sequence length (T)
    Parameters p;    // Parameters (theta)
    HiddenState hs;  // Persistent hidden state updated after a forward pass
    Variables v;     // Allocated memory for calculations
};

static void softmax(Matrix output, float temperature);
static Entry sigmoid(Entry x);
static Entry dsigmoid(Entry x);
static Entry rationalTanh(Entry x);
static Entry drationalTanh(Entry x);

Model ModelNew(int hiddenSize, int numLayers, int seqLength)
{
    Model m = calloc(1, sizeof(struct model));
    if (!m) return NULL;

    m->V = VOCABULARY_SIZE;
    m->E = hiddenSize; // E = H for simplicity
    m->H = hiddenSize;
    m->N = numLayers;
    m->T = seqLength;
    size_t V = m->V, E = m->E, H = m->H, N = m->N, T = m->T;
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

    // Allocate persistent hidden state
    m->hs.h = calloc(N, sizeof(Matrix));
    m->hs.entries = calloc(N * H, sizeof(Entry));
    if (!m->hs.h || !m->hs.entries) goto error;
    addr = m->hs.entries;
    for (size_t i = 0; i < N; i++) {
        m->hs.h[i] = MatrixView(H, 1, addr); addr += H;
    }

    // Allocate variables and assign memory addresses
    m->v.x       = calloc(T * N, sizeof(Matrix));
    m->v.f       = calloc(T * N, sizeof(Matrix));
    m->v.h_hat   = calloc(T * N, sizeof(Matrix));
    m->v.z_h     = calloc(T * N, sizeof(Matrix));
    m->v.h       = calloc(T * N, sizeof(Matrix));
    m->v.y       = calloc(T, sizeof(Matrix));
    m->v.dh      = calloc(N, sizeof(Matrix));
    m->v.h_start = calloc(N, sizeof(Matrix));
    size_t S = V > H ? V : H;
    m->v.entries = calloc(5 * T * N * H + T * V + 2 * N * H + S * H + 3 * H, sizeof(Entry));
    if (!m->v.entries || !m->v.f || !m->v.h_hat || !m->v.x
        || !m->v.z_h || !m->v.h || !m->v.y || !m->v.dh || !m->v.h_start) goto error;
    addr = m->v.entries;
    for (size_t i = 0; i < T * N; i++) {
        m->v.x[i]      = MatrixView(H, 1, addr); addr += H;
        m->v.f[i]      = MatrixView(H, 1, addr); addr += H;
        m->v.h_hat[i]  = MatrixView(H, 1, addr); addr += H;
        m->v.z_h[i]    = MatrixView(H, 1, addr); addr += H;
        m->v.h[i]      = MatrixView(H, 1, addr); addr += H;
    }
    for (size_t i = 0; i < T; i++) {
        m->v.y[i] = MatrixView(V, 1, addr); addr += V;
    }
    m->v.tempV = MatrixView(H, 3, addr); addr += 3 * H;
    m->v.tempM = MatrixView(S, H, addr); addr += S * H;
    for (size_t i = 0; i < N; i++) {
        m->v.dh[i] = MatrixView(H, 1, addr); addr += H;
    }
    for (size_t i = 0; i < N; i++) {
        m->v.h_start[i] = MatrixView(H, 1, addr); addr += H;
    }

    // Allocate Adam optimizer state
    m->v.m = calloc(ModelParameters(m), sizeof(float));
    m->v.v = calloc(ModelParameters(m), sizeof(float));
    m->v.t = 0;
    if (!m->v.m || !m->v.v) goto error;

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
    free(m->v.h);
    free(m->v.y);
    free(m->v.dh);
    free(m->v.h_start);
    free(m->v.m);
    free(m->v.v);
    free(m->hs.h);
    free(m->hs.entries);
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
        return ModelNew(0, 0, 1);
    }

    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    long minSize = 2 * sizeof(size_t) + VOCABULARY_SIZE * sizeof(Entry);
    if (fileSize < minSize) {
        fclose(f);
        return ModelNew(0, 0, 1);
    }
    fseek(f, 0, SEEK_SET);

    size_t H, N;
    fread(&H, sizeof(size_t), 1, f);
    fread(&N, sizeof(size_t), 1, f);
    Model m = ModelNew(H, N, 1);
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
        MatrixZero(m->hs.h[l]);
}

// Accumulates the following variables for backward pass: {x, f, h_hat, z_h, h, y}
Matrix ModelForward(Model m, Token *input)
{
    Variables *v = &m->v;
    size_t T = m->T, N = m->N;
    Matrix tempH1  = MatrixView(m->H, 1, v->tempV.entries);
    Matrix tempH2 = MatrixView(m->H, 1, v->tempV.entries + m->H);

    // Cache pre-forward hidden state for backward pass
    for (size_t l = 0; l < N; l++)
        MatrixCopy(v->h_start[l], m->hs.h[l]);

    for (size_t t = 0; t < T; t++)
    {
        // Embedding layer: x = W_e[input[t], :]^T
        Matrix x = MatrixTranspose(MatrixRow(m->p.e.W, (size_t)input[t]));

        // Iterate over MGU layers
        for (size_t l = 0; l < N; l++)
        {
            size_t index = t * N + l;
            HiddenLayer p = m->p.h[l];
            Matrix h_prev = (t == 0) ? m->hs.h[l] : v->h[(t - 1) * N + l];

            MatrixCopy(v->x[index], x);

            // f = sigmoid(W_f * x + U_f * h_prev + b_f)
            MatrixMultiply(tempH1, p.U_f, h_prev);
            MatrixMultiply(v->f[index], p.W_f, x);
            MatrixAdd(v->f[index], v->f[index], tempH1);
            MatrixAdd(v->f[index], v->f[index], p.b_f);
            MatrixApply(v->f[index], sigmoid);

            // h_hat = tanh(W_h * x + U_h * (f ⊙ h_prev) + b_h)
            MatrixHadamard(tempH2, v->f[index], h_prev);
            MatrixMultiply(tempH1, p.U_h, tempH2);
            MatrixMultiply(v->z_h[index], p.W_h, x);
            MatrixAdd(v->z_h[index], v->z_h[index], tempH1);
            MatrixAdd(v->z_h[index], v->z_h[index], p.b_h);
            MatrixCopy(v->h_hat[index], v->z_h[index]);
            MatrixApply(v->h_hat[index], rationalTanh);

            // h = (1 - f) ⊙ h_prev + f ⊙ h_hat
            MatrixHadamard(tempH1, v->f[index], h_prev);
            MatrixSubtract(v->h[index], h_prev, tempH1);
            MatrixHadamard(tempH1, v->f[index], v->h_hat[index]);
            MatrixAdd(v->h[index], v->h[index], tempH1);

            // Next layer input is the previous layer output
            x = v->h[index];
        }

        // Output layer: y = W_o * x + b_o
        MatrixMultiply(v->y[t], m->p.o.W, x);
        MatrixAdd(v->y[t], v->y[t], m->p.o.b);
    }

    // Advance hidden state
    for (size_t l = 0; l < N; l++)
        MatrixCopy(m->hs.h[l], v->h[(T - 1) * N + l]);

    return v->y[T - 1];
}

// Accumulates gradients (dL/dtheta)
float ModelBackward(Model m, Token *input, Token *target)
{
    Variables *v = &m->v;
    size_t V = m->V, H = m->H, N = m->N, T = m->T;
    Matrix tempH1  = MatrixView(H, 1, v->tempV.entries);         // Temporary vector [H x 1]
    Matrix tempH2  = MatrixView(H, 1, v->tempV.entries + H);     // Temporary vector [H x 1]
    Matrix tempH3  = MatrixView(H, 1, v->tempV.entries + 2 * H); // Temporary vector [H x 1]
    Matrix tempVxH = MatrixView(V, H, v->tempM.entries);         // Temporary matrix [V x H]
    Matrix tempHxH = MatrixView(H, H, v->tempM.entries);         // Temporary matrix [H x H]
    for (size_t l = 0; l < N; l++) MatrixZero(v->dh[l]);
    float loss = 0.0f;

    for (size_t t = T; t-- > 0; )
    {
        // Loss = -log(softmax(y)[target])
        softmax(v->y[t], 1.0f);
        loss += -logf(MatrixGet(v->y[t], target[t], 0));

        // dL/dy = softmax(y) - one_hot
        MatrixSet(v->y[t], target[t], 0, MatrixGet(v->y[t], target[t], 0) - 1.0f);

        // Output layer
        // dL/dW_o += dL/dy * h^T
        MatrixMultiply(tempVxH, v->y[t], MatrixTranspose(v->h[t * N + N - 1]));
        MatrixAdd(v->dp.o.W, v->dp.o.W, tempVxH);

        // dL/db_o += dL/dy
        MatrixAdd(v->dp.o.b, v->dp.o.b, v->y[t]);

        // dL/dh += W_o^T * dL/dy
        MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(v->y[t]), m->p.o.W);
        MatrixAdd(v->dh[N - 1], v->dh[N - 1], tempH1);

        // Iterate over MGU layers backwards
        for (size_t l = N; l-- > 0; )
        {
            HiddenLayer p = m->p.h[l];   // Hidden layer parameters
            HiddenLayer dp = v->dp.h[l]; // Hidden layer gradients
            size_t index = t * N + l;
            Matrix h_prev = (t == 0) ? v->h_start[l] : v->h[(t - 1) * N + l];

            // dL/dz_h = (dL/dh ⊙ f) ⊙ tanh'(z_h)
            MatrixCopy(tempH3, v->z_h[index]);
            MatrixApply(tempH3, drationalTanh);
            MatrixHadamard(tempH1, v->dh[l], v->f[index]);
            MatrixHadamard(tempH3, tempH1, tempH3);

            // dL/dW_h += dL/dz_h * x^T
            MatrixMultiply(tempHxH, tempH3, MatrixTranspose(v->x[index]));
            MatrixAdd(dp.W_h, dp.W_h, tempHxH);

            // dL/dU_h += dL/dz_h * (f ⊙ h_prev)^T
            MatrixHadamard(tempH1, v->f[index], h_prev);
            MatrixMultiply(tempHxH, tempH3, MatrixTranspose(tempH1));
            MatrixAdd(dp.U_h, dp.U_h, tempHxH);

            // dL/db_h += dL/dz_h
            MatrixAdd(dp.b_h, dp.b_h, tempH3);

            // dL/df = dL/dh ⊙ (h_hat - h_prev) + (U_h^T * dL/dz_h) ⊙ h_prev
            //         dL/dh ⊙ (h_hat - h_prev) + ((dL/dz_h)^T * U_h)^T ⊙ h_prev
            MatrixSubtract(tempH1, v->h_hat[index], h_prev);
            MatrixHadamard(tempH2, v->dh[l], tempH1);
            MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(tempH3), p.U_h);
            MatrixHadamard(tempH1, tempH1, h_prev);
            MatrixAdd(tempH2, tempH2, tempH1);

            // dL/dz_f = dL/df ⊙ sigmoid'(f)
            MatrixCopy(tempH1, v->f[index]);
            MatrixApply(tempH1, dsigmoid);
            MatrixHadamard(tempH2, tempH2, tempH1);

            // dL/dW_f += dL/dz_f * x^T
            MatrixMultiply(tempHxH, tempH2, MatrixTranspose(v->x[index]));
            MatrixAdd(dp.W_f, dp.W_f, tempHxH);

            // dL/dU_f += dL/dz_f * h_prev^T
            MatrixMultiply(tempHxH, tempH2, MatrixTranspose(h_prev));
            MatrixAdd(dp.U_f, dp.U_f, tempHxH);

            // dL/db_f += dL/dz_f
            MatrixAdd(dp.b_f, dp.b_f, tempH2);

            // dL/dh = dL/dh ⊙ (1-f) + (U_h^T * dL/dz_h) ⊙ f + U_f^T * dL/dz_f
            MatrixHadamard(tempH1, v->dh[l], v->f[index]);
            MatrixSubtract(v->dh[l], v->dh[l], tempH1);
            MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(tempH3), p.U_h);
            MatrixHadamard(tempH1, tempH1, v->f[index]);
            MatrixAdd(v->dh[l], v->dh[l], tempH1);
            MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(tempH2), p.U_f);
            MatrixAdd(v->dh[l], v->dh[l], tempH1);

            // dL/dx = W_h^T * dL/dz_h + W_f^T * dL/dz_f
            //       = ((dL/dz_h)^T * W_h)^T + ((dL/dz_f)^T * W_f)^T
            MatrixMultiply(MatrixTranspose(tempH1), MatrixTranspose(tempH3), p.W_h);
            MatrixMultiply(MatrixTranspose(tempH3), MatrixTranspose(tempH2), p.W_f);
            MatrixAdd(tempH1, tempH1, tempH3);

            // dL/dh += dL/dx
            if (l > 0) MatrixAdd(v->dh[l - 1], v->dh[l - 1], tempH1);
        }
        // Embedding layer
        // dL/dW_e[input[t], :] += dL/dx
        Matrix dW_e = MatrixTranspose(MatrixRow(v->dp.e.W, (size_t)input[t]));
        MatrixAdd(dW_e, dW_e, tempH1);
    }

    return loss;
}

// Adam optimizer with gradient clipping
void ModelAdam(Model m, float learningRate)
{
    size_t n = ModelParameters(m);
    float *p = m->p.entries;
    float *dp = m->v.dp.entries;

    float norm = 0.0f;
    for (size_t i = 0; i < n; i++)
        norm += dp[i] * dp[i];
    norm = sqrtf(norm);
    if (!isfinite(norm)) {
        memset(dp, 0, n * sizeof(float));
        return;
    }
    float scale = (norm > CLIP) ? CLIP / norm : 1.0f;

    // Adam update
    m->v.t++;
    float b1 = 1.0f - powf(BETA1, m->v.t);
    float b2 = 1.0f - powf(BETA2, m->v.t);
    for (size_t i = 0; i < n; i++) {
        float g = dp[i] * scale;
        m->v.m[i] = BETA1 * m->v.m[i] + (1.0f - BETA1) * g;
        m->v.v[i] = BETA2 * m->v.v[i] + (1.0f - BETA2) * g * g;
        float m_hat = m->v.m[i] / b1;
        float v_hat = m->v.v[i] / b2;
        p[i] -= learningRate * m_hat / (sqrtf(v_hat) + EPSILON);
        dp[i] = 0.0f;
    }
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
