#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Model.h"

// Number of matrices/biases in a GRU cell
#define NUM_WEIGHT_MATRICES 6
#define NUM_BIAS_VECTORS 3

// A GRU model with an embedding layer - https://en.wikipedia.org/wiki/Gated_recurrent_unit
struct model
{
    int numLayers;      // Number of hidden layers
    int hiddenSize;     // Size of the hidden layer(s)
    float *weights;     // Flattened array of weights
    float *hiddenState; // Flattened array of hidden states for each GRU layer
};

static uint32_t xorshift();
static void xavierInitialization(float *weights, int rows, int cols);
static float randomFloat(float bound);

Model ModelNew(int numLayers, int hiddenSize)
{
    Model m = malloc(sizeof(struct model));
    if (m == NULL)
    {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    m->numLayers = numLayers;
    m->hiddenSize = hiddenSize;

    // Calculate the total number of weights/biases in the model
    size_t embeddingSize = VOCABULARY_SIZE * hiddenSize;                // W_e
    size_t gruLayerSize = hiddenSize * hiddenSize * NUM_WEIGHT_MATRICES // W_z, W_r, W_h, U_z, U_r, U_h
                                     + hiddenSize * NUM_BIAS_VECTORS;   // b_z, b_r, b_h
    size_t outputSize = hiddenSize * VOCABULARY_SIZE + VOCABULARY_SIZE; // W_o, b_o
    size_t totalWeights = embeddingSize + gruLayerSize * numLayers + outputSize;

    // Allocate memory for weights/biases and initialzie to 0
    m->weights = calloc(totalWeights, sizeof(float));
    if (m->weights == NULL)
    {
        free(m);
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weighs with Xavier initialization, leave biases as 0
    size_t offset = 0;

    // W_e
    xavierInitialization(m->weights + offset, VOCABULARY_SIZE, hiddenSize);
    offset += embeddingSize;

    for (int l = 0; l < numLayers; l++)
    {
        for (int i = 0; i < NUM_WEIGHT_MATRICES; i++)
        {
            // W_z, W_r, W_h, U_z, U_r, U_h
            xavierInitialization(m->weights + offset, hiddenSize, hiddenSize);
            offset += hiddenSize * hiddenSize;
        }
        offset += hiddenSize * NUM_BIAS_VECTORS; // Skip b_z, b_r, b_h
    }

    // W_o
    xavierInitialization(m->weights + offset, hiddenSize, VOCABULARY_SIZE);
    offset += hiddenSize * VOCABULARY_SIZE;

    // Allocate memory for the hidden states and initialize to 0
    m->hiddenState = calloc(numLayers * hiddenSize, sizeof(float));
    if (m->hiddenState == NULL)
    {
        free(m->weights);
        free(m);
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    return m;
}

void ModelFree(Model m)
{
    free(m->hiddenState);
    free(m->weights);
    free(m);
}

Model ModelRead(char *path)
{
    return ModelNew(0, 0);
}

void ModelWrite(Model m, char *path)
{
    return;
}

void ModelReset(Model m)
{
    return;
}

void ModelForward(Model m, Token input, Logits output)
{
    return;
}

Token ModelSample(Model m, Logits output)
{
    return 0;
}

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

// Returns a random float in [-bound, +bound]
static inline float randomFloat(float bound)
{
    return ((xorshift() >> 8) * (2.0f / 16777216.0f) - 1.0f) * bound;
}

// Xavier initialize a matrix of weights
static void xavierInitialization(float *weights, int rows, int cols) {
    size_t size = rows * cols;
    float bound = sqrtf(6.0f / (rows + cols));
    for (size_t i = 0; i < size; i++) {
        weights[i] = randomFloat(bound);
    }
}

