// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   stderr:
//     --- Begin aot ---
//     ...
//     %{{_}}: i64 = promote %{{_}} [safepoint: ...
//     ...
//     %{{_}}: i64 = promote %{{_}} [safepoint: ...
//     ...
//     %{{_}}: i64 = promote %{{_}} [safepoint: ...
//     ...
//     %{{_}}: i64 = promote %{{_}} [safepoint: ...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i1 = eq %{{_}}, 100i64
//     guard true, %{{1}}, ...
//     ...
//     %{{2}}: i1 = eq %{{_}}, 99i64
//     guard true, %{{2}}, ...
//     ...
//     %{{3}}: i1 = eq %{{_}}, 88i64
//     guard true, %{{3}}, ...
//     ...
//     %{{5}}: i1 = eq %{{_}}, 2i64
//     guard true, %{{5}}, ...
//     ...
//     --- End jit-pre-opt ---

// Check that promotion works in traces.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

uint64_t inner(uint64_t a, uint64_t b, uint64_t x, uint64_t y, uint64_t z) {
  a = yk_promote(a);
  b = yk_promote(b);
  x = yk_promote(x);
  z = yk_promote(z);
  y += x * 3 * z;  // +1
  y += a * 10 * b; // no-op
  return y;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  uint64_t a = 100, b = 99, x = 88, y = 0, z = 2;
  NOOPT_VAL(a);
  NOOPT_VAL(b);
  NOOPT_VAL(x);
  NOOPT_VAL(z);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    y = inner(a, b, x, y, z);
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
