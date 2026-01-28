#include <stdio.h>
#include <stdlib.h>
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

    // Calculate the total number of weights in the model
    size_t embeddingSize = VOCABULARY_SIZE * hiddenSize;                // W_e
    size_t gruLayerSize = hiddenSize * hiddenSize * NUM_WEIGHT_MATRICES // W_z, W_r, W_h, U_z, U_r, U_h
                                     + hiddenSize * NUM_BIAS_VECTORS;   // b_z, b_r, b_h
    size_t outputSize = hiddenSize * VOCABULARY_SIZE + VOCABULARY_SIZE; // W_o, b_o
    size_t totalWeights = embeddingSize + gruLayerSize * numLayers + outputSize;

    // Allocate memory for weights
    m->weights = malloc(totalWeights * sizeof(float));
    if (m->weights == NULL)
    {
        free(m);
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weights with Xavier initialization - https://en.wikipedia.org/wiki/Weight_initialization
    // TODO

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
