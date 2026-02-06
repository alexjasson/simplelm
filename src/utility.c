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
