// Trace decoder benchmarks.
//
// Criterion forks executions of this code.

#define _GNU_SOURCE

#include <err.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

#define MAX_TRACE_PATH 64

#define BM_NATIVE 0
#define BM_DISASM 1

// The number of temporary failures we will tolerate before failing.
#define MAX_TRIES 3

__attribute__((noinline)) uint64_t native(uint64_t iters) {
  uint64_t sum = 0;
  NOOPT_VAL(iters);
  while (iters) {
    sum += iters--;
  }
  NOOPT_VAL(sum);
  return sum;
}

__attribute__((noinline)) uint64_t disasm(uint64_t iters) {
  char *buf = malloc(0);
  NOOPT_VAL(iters);
  for (; iters != 0; iters--) {
    char *tmp;
    if (asprintf(&tmp, "%s %lu\n", buf, iters) < 0) {
      err(EXIT_FAILURE, "asprintf");
    }
    free(buf);
    buf = tmp;
  }

  size_t res = strlen(buf);
  free(buf);
  NOOPT_VAL(res);
  return res;
}

// Returns `true` on success, or `false` if the benchmark needs to be re-tried.
bool collect_and_decode(int benchmark, size_t param) {
  uint64_t res;

  void *tc = __hwykpt_start_collector();
  if (benchmark == BM_NATIVE) {
    res = native(param);
  } else if (benchmark == BM_DISASM) {
    res = disasm(param);
  } else {
    errx(EXIT_FAILURE, "unreachable");
  }
  void *trace = __hwykpt_stop_collector(tc);
  NOOPT_VAL(res);

  return __hwykpt_decode_trace(trace);
}

void usage(void) {
  printf("args: <benchmark> <param>\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc != 3)
    usage();

  for (int i = MAX_TRIES; i > 0; i--) {
    if (collect_and_decode(atoi(argv[1]), atoi(argv[2]))) {
      return EXIT_SUCCESS;
    }
  }

  fprintf(stderr, "fatal: exceeded max re-tries\n");
  return EXIT_FAILURE;
}
