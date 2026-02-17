// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   stderr:
//     yk-tracing: start-tracing
//     a=99 b=765 y=100
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{_}}: i64 = promote %{{_}} ...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{8}}: i32 = 99
//     %{{9}}: i1 = icmp eq %{{_}}, %{{8}}
//     guard true, %{{9}}, ...
//     ...
//     %{{13}}: i64 = 765
//     %{{14}}: i1 = icmp eq %{{_}}, %{{13}}
//     guard true, %{{14}}, ...
//     ...
//     %{{18}}: i64 = 100
//     %{{19}}: i1 = icmp eq %{{_}}, %{{18}}
//     guard true, %{{19}}, ...
//     ...
//     --- End jit-pre-opt ---
//     a=99 b=765 y=200
//     yk-execution: enter-jit-code
//     a=99 b=765 y=300
//     a=99 b=765 y=400
//     a=99 b=765 y=500
//     yk-execution: deoptimise ...

// Check that promotion works in traces.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int a = 99;
  long long b = 765;
  size_t x = 100;
  size_t y = 0;
  NOOPT_VAL(a);
  NOOPT_VAL(b);
  NOOPT_VAL(x);
  NOOPT_VAL(y);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    a = yk_promote(a);
    b = yk_promote(b);
    x = yk_promote(x);
    y += x;
    fprintf(stderr, "a=%d b=%lld y=%" PRIu64 "\n", a, b, y);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
