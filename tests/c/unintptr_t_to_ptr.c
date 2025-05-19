// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 0x4
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{11_2}}: ptr = int_to_ptr %{{11_1}}, ptr
//     ...
//     %{{11_3}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{11_1}}, %{{11_2}})
//     ...
//     --- End aot ---
//     3: 0x3
//     yk-execution: enter-jit-code
//     2: 0x2
//     1: 0x1
//     yk-execution: deoptimise

// Check that converting an integer to a pointer works.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  uintptr_t i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%" PRIuPTR ": %p\n", i, (void *) i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
