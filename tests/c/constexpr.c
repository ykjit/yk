// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-tracing: start-tracing
//     100
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{_}}: i32 = call first(const_gep(@.str.1+3))...
//     ...
//     --- End aot ---
//     100
//     yk-execution: enter-jit-code {"trid": "0"}
//     100
//     100
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     exit

// Constant expressions can refer to private globals with no dynamic symbol.
// Resolve their addresses through the compiler-emitted global pointer array.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int first(const char *str) {
  NOOPT_VAL(str);
  return str[0];
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d\n", first(&"abcd"[3]));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return EXIT_SUCCESS;
}
