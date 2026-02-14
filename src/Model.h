#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "Matrix.h"

#define VOCABULARY_SIZE 256 // All ASCII characters

typedef struct model *Model;
typedef uint8_t Token; // Input of the model

/*
 * Allocates memory for a model on the heap.
 */
Model ModelNew(int hiddenSize, int numLayers, int seqLength);

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
 * Given an array of T input tokens, update the output vectors [V x 1] with the
 * result of a forward pass through the model. Accumulates variables at each
 * timestep and advances the persistent hidden state.
 */
Matrix ModelForward(Model m, Token *input);

/*
 * Given the arrays of T input/target tokens, accumulate gradients (dLoss/dtheta)
 * and return the total cross-entropy loss over T timesteps. Must be called after
 * ModelForward.
 */
float ModelBackward(Model m, Token *input, Token *target);

/*
 * Given the output of the model, choose a token and return it.
 */
Token ModelSample(Model m, Matrix output, float temperature);

#endif