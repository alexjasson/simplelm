#ifndef THREADPOOL_H
#define THREADPOOL_H

typedef struct threadPool *ThreadPool;
typedef void (*MapFunction)(void *arg, size_t start, size_t end);

/*
 * Singleton pattern to initialize thread pool once then free on exit.
 * Number of threads set to number of cores in CPU.
 */
ThreadPool ThreadPoolGet(void);

/*
 * Divides total items into subsets, maps f across total threads and blocks until complete
 */
void ThreadPoolMap(ThreadPool pool, MapFunction f, void *arg, size_t total);

#endif
