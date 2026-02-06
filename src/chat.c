/*
 * SimpleLM
 * © 2026 Alex Jasson
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Model.h"

#define MODEL "data/model.out"
#define MAX_INPUT 1024

int main(int argc, char **argv)
{

    // Check arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <temperature>\n", argv[0]);
        return 1;
    }

    float temperature = atof(argv[1]);
    Model m = ModelRead(MODEL);
    Matrix output = MatrixNew(VOCABULARY_SIZE, 1);
    char input[MAX_INPUT];

    while (1) {
        printf("You: ");
        if (!fgets(input, MAX_INPUT, stdin)) break;

        for (size_t i = 0; i < strlen(input); i++)
            ModelForward(m, (Token)input[i], output);

        printf("Model: ");
        Token t;
        while ((t = ModelSample(m, output, temperature)) != '\n') {
            putchar(t);
            fflush(stdout);
            ModelForward(m, t, output);
        }
        printf("\n");
    }

    MatrixFree(output);
    ModelFree(m);
    return 0;
}