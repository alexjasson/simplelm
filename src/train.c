/*
 * SimpleLM
 * © 2026 Alex Jasson
 */

#include <stdio.h>
#include <stdlib.h>
#include "Model.h"
#include "utility.h"

#define MODEL "data/model.out"
#define DATA  "data/alphabet.in"
#define WRITE 1000
#define LOG   100

int main(int argc, char **argv)
{
    // Check arguments
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <hiddenSize> <numLayers> <seqLength>\n", argv[0]);
        return 1;
    }

    size_t hiddenSize = atoi(argv[1]);
    size_t numLayers  = atoi(argv[2]);
    size_t seqLength  = atoi(argv[3]);
    size_t dataSize = fileSize(DATA);
    if (dataSize < 1) return 0;
    if (dataSize - 1 < seqLength) seqLength = dataSize - 1;

    Token *data = malloc(dataSize);
    if (!data) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    readFile(DATA, data);
    Model m = ModelNew(hiddenSize, numLayers, seqLength);
    printf("Parameters: %zu\n", ModelParameters(m));

    // Basic training loop for 1 epoch
    size_t offset = 0;
    size_t totalSteps = dataSize / seqLength;
    for (size_t step = 1; step <= totalSteps; step++) {
        Token *input  = &data[offset];
        Token *target = &data[offset + 1];

        ModelForward(m, input);
        float loss = ModelBackward(m, input, target);
        ModelOptimizer(m);

        offset += seqLength;
        if (step % LOG == 0)
            printf("Step %zu: loss %.4f\n", step, loss);

        if (step % WRITE == 0) {
            ModelWrite(m, MODEL);
            printf("Model saved to %s\n", MODEL);
        }
    }

    ModelFree(m);
    free(data);
    return 0;
}
