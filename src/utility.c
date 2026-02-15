#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "utility.h"

static uint32_t xorshift(void)
{
    static uint32_t state = 0;
    static int seeded = 0;

    if (!seeded) {
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

float randomFloat(float lower, float upper)
{
    float r = (xorshift() >> 8) / 16777216.0f;
    return lower + r * (upper - lower);
}

size_t fileSize(char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fclose(f);
    return size;
}

void readFile(char *path, void *data)
{
    size_t length = fileSize(path);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    fread(data, 1, length, f);
    fclose(f);
}
