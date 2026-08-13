// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     foo 337
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func g(...
//     ...
//     #[yk_indirect_inline]
//     func f(...
//     ...
//     %{{1_0}}: i32 = call g(...
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{333}}: i32 = 333
//     %{{_}}: i32 = add %{{_}}, %{{333}}
//     ...
//     --- End hir ---
//     foo 336
//     yk-execution: enter-jit-code {"trid": "0"}
//     foo 335
//     foo 334
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that a function only reachable from a `yk_indirect_inline` function is
// considered inlinable into a trace.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int g(int i) {
  return i + 333;
}

__attribute__((yk_indirect_inline))
int f(int i) {
  return g(i);
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  int (*fn)(int) = f;

  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(fn);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int (*fp)(int) = yk_promote((void *)fn);
    fprintf(stderr, "foo %d\n", fp(i));
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return EXIT_SUCCESS;
}
