#include <stdio.h>
#include <stdlib.h>
#include "Model.h"
#include "utility.h"

#define MODEL "data/model.out"
#define DATA  "data/dialogue.in"
#define LOG   100

int main(int argc, char **argv)
{
    // Check arguments
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <hiddenSize> <numLayers> <seqLength>\n", argv[0]);
        return 1;
    }
    if (fileSize(DATA) <= atoi(argv[3])) {
        fprintf(stderr, "Training data must be larger than sequence length\n", argv[0]);
        return 1;
    }

    size_t hiddenSize = atoi(argv[1]);
    size_t numLayers  = atoi(argv[2]);
    size_t seqLength  = atoi(argv[3]);
    Token *data = readFile(DATA);
    Model m = ModelNew(hiddenSize, numLayers, seqLength);
    printf("Training model with %zu parameters\n", ModelParameters(m));

    // Basic training loop for 1 epoch
    size_t offset = 0;
    size_t totalSteps = fileSize(DATA)  / seqLength;
    for (size_t step = 1; step <= totalSteps; step++) {
        Token *input  = &data[offset];
        Token *target = &data[offset + 1];

        ModelForward(m, input);
        float loss = ModelBackward(m, input, target);
        ModelOptimizer(m);

        offset += seqLength;
        if (step % LOG == 0) printf("Step %zu: loss %.4f\n", step, loss);
    }

    ModelWrite(m, MODEL);
    printf("Model saved to %s\n", MODEL);
    ModelFree(m);
    free(data);
    return 0;
}
