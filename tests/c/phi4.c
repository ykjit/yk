// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=aot
//   stderr:
//     yk-tracing: start-tracing
//     0: 3
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{_}}: i32 = phi bb{{back}} -> %{{_}}, bb{{entry}} -> %{{_}}
//     %{{a}}: i32 = phi bb{{entry}} -> %{{_}}, bb{{back}} -> %{{b}}
//     %{{b}}: i32 = phi bb{{entry}} -> %{{_}}, bb{{back}} -> %{{a}}
//     ...
//     --- End aot ---
//     1: 3
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 3
//     3: 3
//     yk-execution: deoptimise ...
//     exit

// Check that PHIs are processed as if they happen simultaneously rather than
// one-at-a-time.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_unroll, noinline)) int swap(int n, int a, int b) {
  for (int i = 0; i < n; i++) {
    int tmp = a;
    a = b;
    b = tmp;
  }
  return a + b;
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int n = 3;
  int a = 1;
  int b = 2;
  NOOPT_VAL(n);
  NOOPT_VAL(a);
  NOOPT_VAL(b);
  for (int i = 0; i < 4; i++) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %d\n", i, swap(n, a, b));
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return EXIT_SUCCESS;
}
