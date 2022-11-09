#ifndef __HWTRACER_UTIL_H
#define __HWTRACER_UTIL_H

#include <err.h>

/*
 * Crash out with a formatted message.
 */
#define panic(...)                                              \
    fprintf(stderr, "Panic at %s:%d: ", __FILE__, __LINE__);    \
    errx(EXIT_FAILURE, __VA_ARGS__);

/*
 * Print debug messages to stderr.
 * For development only.
 */
#define DEBUG(x...)                       \
    do {                                  \
        fprintf(stderr, "%s:%d [%s]: ",   \
           __FILE__, __LINE__, __func__); \
        fprintf(stderr, x);               \
        fprintf(stderr, "\n");            \
    } while (0)

#endif
