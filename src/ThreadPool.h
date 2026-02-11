#ifndef THREADPOOL_H
#define THREADPOOL_H

typedef struct threadPool *ThreadPool;
typedef void (*MapFunction)(void *arg, size_t start, size_t end);

/*
 * Creates a thread pool with the given number of worker threads.
 */
ThreadPool ThreadPoolNew(size_t numThreads);

/*
 * Divides total items into subsets, maps f across numThreads and blocks until complete
 */
void ThreadPoolMap(ThreadPool pool, MapFunction f, void *arg, size_t total);

/*
 * Destroys the thread pool and joins all worker threads.
 */
void ThreadPoolFree(ThreadPool pool);

#endif
