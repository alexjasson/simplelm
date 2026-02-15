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

    Model m = ModelRead(MODEL);
    float temperature = atof(argv[1]);
    char input[MAX_INPUT];

    while (1) {
        printf("You: ");
        if (!fgets(input, MAX_INPUT, stdin)) break;

        Matrix output;
        for (size_t i = 0; i < strlen(input); i++) {
            Token t = (Token)input[i];
            output = ModelForward(m, &t);
        }

        printf("Model: ");
        Token t;
        while ((t = ModelSample(m, output, temperature)) != '\n') {
            putchar(t);
            fflush(stdout);
            output = ModelForward(m, &t);
        }
        printf("\n");
    }

    ModelFree(m);
    return 0;
}
