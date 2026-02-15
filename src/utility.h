#ifndef UTILITY_H
#define UTILITY_H

/*
 * Returns a random float in [lower, upper)
 */
float randomFloat(float lower, float upper);

/*
 * Returns the size of the given file in bytes
 */
size_t fileSize(char *path);

/*
 * Reads the contents of a file into the given data array
 */
void readFile(char *path, void *data);

#endif
