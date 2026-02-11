#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "ThreadPool.h"

typedef struct {
    ThreadPool pool;
    size_t id;
} ThreadArguments;

struct threadPool {
    pthread_t *threads;
    size_t numThreads;
    ThreadArguments *threadArgs;

    pthread_mutex_t mutex;
    pthread_cond_t mapReady;
    pthread_cond_t mapDone;

    MapFunction f;
    void *mapArgs;
    size_t mapTotal;
    size_t mapId;
    size_t subsetsDone;
    int shutdown;
};


static void *threadFn(void *arg)
{
    ThreadArguments *ta = arg;
    ThreadPool pool = ta->pool;
    size_t id = ta->id;
    size_t lastMapId = 0;

    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->mapId == lastMapId && !pool->shutdown)
            pthread_cond_wait(&pool->mapReady, &pool->mutex);

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }

        lastMapId = pool->mapId;
        MapFunction f = pool->f;
        void *ma = pool->mapArgs;
        size_t total = pool->mapTotal;
        size_t n = pool->numThreads;
        pthread_mutex_unlock(&pool->mutex);

        size_t subsetSize = total / n;
        size_t remainder = total % n;
        size_t start = id * subsetSize + (id < remainder ? id : remainder);
        size_t end = start + subsetSize + (id < remainder ? 1 : 0);

        if (start < end)
            f(ma, start, end);

        pthread_mutex_lock(&pool->mutex);
        pool->subsetsDone++;
        if (pool->subsetsDone == n)
            pthread_cond_signal(&pool->mapDone);
        pthread_mutex_unlock(&pool->mutex);
    }
}

ThreadPool ThreadPoolNew(size_t n)
{
    ThreadPool pool = malloc(sizeof(struct threadPool));
    if (!pool) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    pool->numThreads = n;
    pool->mapId = 0;
    pool->shutdown = 0;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->mapReady, NULL);
    pthread_cond_init(&pool->mapDone, NULL);

    pool->threads = malloc(n * sizeof(pthread_t));
    pool->threadArgs = malloc(n * sizeof(ThreadArguments));
    if (!pool->threads || !pool->threadArgs) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < n; i++) {
        pool->threadArgs[i] = (ThreadArguments){pool, i};
        pthread_create(&pool->threads[i], NULL, threadFn, &pool->threadArgs[i]);
    }

    return pool;
}

void ThreadPoolMap(ThreadPool pool, MapFunction f, void *arg, size_t total)
{
    pthread_mutex_lock(&pool->mutex);
    pool->f = f;
    pool->mapArgs = arg;
    pool->mapTotal = total;
    pool->subsetsDone = 0;
    pool->mapId++;
    pthread_cond_broadcast(&pool->mapReady);

    while (pool->subsetsDone < pool->numThreads)
        pthread_cond_wait(&pool->mapDone, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

void ThreadPoolFree(ThreadPool pool)
{
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->mapReady);
    pthread_mutex_unlock(&pool->mutex);

    for (size_t i = 0; i < pool->numThreads; i++)
        pthread_join(pool->threads[i], NULL);

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->mapReady);
    pthread_cond_destroy(&pool->mapDone);
    free(pool->threadArgs);
    free(pool->threads);
    free(pool);
}
