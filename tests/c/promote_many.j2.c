// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   stderr:
//     yk-tracing: start-tracing
//     99528
//     yk-tracing: stop-tracing
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
//     %{{30}}: i64 = 100
//     %{{31}}: i1 = icmp eq %{{_}}, %{{30}}
//     guard true, %{{31}}, ...
//     ...
//     %{{35}}: i64 = 99
//     %{{36}}: i1 = icmp eq %{{_}}, %{{35}}
//     guard true, %{{36}}, ...
//     ...
//     %{{40}}: i64 = 88
//     %{{41}}: i1 = icmp eq %{{_}}, %{{40}}
//     guard true, %{{41}}, ...
//     ...
//     %{{45}}: i64 = 2
//     %{{46}}: i1 = icmp eq %{{_}}, %{{45}}
//     guard true, %{{46}}, ...
//     ...
//     --- End jit-pre-opt ---
//     199056
//     yk-execution: enter-jit-code
//     298584
//     398112
//     497640
//     yk-execution: deoptimise ...
//     exit

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
    fprintf(stderr, "%lu\n", y);
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit");

  return (EXIT_SUCCESS);
}
