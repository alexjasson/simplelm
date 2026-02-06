#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#define VOCABULARY_SIZE 256 // All ASCII characters

typedef float Entry;
typedef struct {
    size_t rows;
    size_t cols;
    Entry *entries;
} Matrix;
typedef float (*MatrixFunction)(Entry);

typedef struct model *Model;
typedef uint8_t Token; 	    // Input of the model

/*
 * Allocates memory for a model on the heap.
 */
Model ModelNew(int hiddenSize, int numLayers);

/*
 * Frees the model from memory.
 */
void ModelFree(Model m);

/*
 * Given a filepath, read the model from disk and allocate memory for it on
 * the heap. If the file contains less bytes than what is possible for a minimum
 * size model then it returns a dummy model with numLayers=0 and hiddenSize=0.
 */
Model ModelRead(char *path);

/*
 * Given a filepath, write the model from RAM to disk. Note that any model
 * previously in the file will be overwritten.
 */
void ModelWrite(Model m, char *path);

/*
 * Returns the total number of parameters in the model.
 */
size_t ModelParameters(Model m);

/*
 * Resets the models memory (hidden state)
 */
void ModelReset(Model m);

/*
 * Given an input token, update the output vector [V x 1] with the result 
 * of a forward pass through the model.
 */
void ModelForward(Model m, Token input, Matrix output);

/*
 * Given the output of the model, choose a token and return it.
 */
Token ModelSample(Model m, Matrix output);

#endif