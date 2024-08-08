#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

uint64_t inner(uint64_t a, uint64_t b, uint64_t x, uint64_t y, uint64_t z) {
#ifdef DO_PROMOTE
  yk_promote(a);
  yk_promote(b);
  yk_promote(x);
  yk_promote(z);
#endif
  y += x * 3 - z;  // +1
  y += a - 10 * b; // no-op
  return y;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: promote <reps>\n");
    exit(EXIT_FAILURE);
  }
  int reps = atoi(argv[1]);

  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  uint64_t a = 100, b = 10, x = 1, y = 0, z = 2;
  NOOPT_VAL(a);
  NOOPT_VAL(b);
  NOOPT_VAL(x);
  NOOPT_VAL(z);

  for (int i = 0; i < reps; i++) {
    yk_mt_control_point(mt, &loc);
    y = inner(a, b, x, y, z);
  }

  assert(y == reps);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
