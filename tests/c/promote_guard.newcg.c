// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     y=100
//     y=200
//     y=300
//     y=400
//     y=500
//     y=700
//     y=800
//     y=900
//     y=1000
//     y=1999

// Check that promotions are guarded correctly.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define ELEMS 10

size_t inner(size_t x, size_t y) {
  yk_promote(x);
  y += x;
  return y;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  // We will trace 100 baked into the trace, and every iteration where there's
  // a 200 we should guard fail.
  size_t xs[ELEMS] = {100, 100, 100, 100, 100, 200, 100, 100, 100, 999};
  size_t y = 0;
  NOOPT_VAL(xs);

  // On the third iteration, a guard must fail to stop us blindly adding 100
  // instead of 200.
  for (int i = 0; i < ELEMS; i++) {
    yk_mt_control_point(mt, &loc);
    y = inner(xs[i], y);
    fprintf(stderr, "y=%" PRIu64 "\n", y);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
