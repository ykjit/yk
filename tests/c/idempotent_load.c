// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 97 97
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{11_1}}: i8 = call idempotent __yk_idempotent_load_uint8_t(%{{_}})
//     ...
//     %{{17_2}}: i32 = zext %{{11_1}}, i32
//     ...
//     %{{_}}: i32 = call fprintf(%{{_}}, @.str.1, %{{_}}, %{{17_2}}, %{{_}})
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{13}}: i32 = load %1
//     %{{14}}: i32 = zext %{{5}}
//     %{{15}}: i32 = 97
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{13}}, %{{14}}, %{{15}}) ; @fprintf
//     ...
//     --- End hir ---
//     3: 97 97
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 97 97
//     1: 97 97
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}

// Check that yk_idempotent_load works as expected.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  uint8_t *bytes = (uint8_t *) "abcd";

  int i = 4;
  uint8_t *m = bytes;
  NOOPT_VAL(i);
  NOOPT_VAL(loc);
  NOOPT_VAL(m);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    uint8_t b1 = yk_idempotent_load(m);
    uint8_t *p = yk_promote(m);
    uint8_t b2 = yk_idempotent_load(p);
    fprintf(stderr, "%" PRIu8 ": %" PRIu8  " %" PRIu8  "\n", i, b1, b2);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
