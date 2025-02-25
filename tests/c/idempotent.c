// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4: 39 39
//     yk-jit-event: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_idempotent, yk_outline]
//     func add(...
//     ...
//     func main(...
//     ...
//     %{{_}}: i{{size}} = idempotent_promote %{{_}}
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i{{size}} = call @add(%{{_}}, 2i{{size}}) <idem_const 39i{{size}}>
//     ...
//     %{{_}}: i{{size}} = call @add(%{{_}}, 2i{{size}}) <idem_const 39i{{size}}>
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     %{{_}}: i{{size}} = call @add(%{{_}}, 2i{{size}}) <idem_const 39i{{size}}>
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, %{{_}}, 39i{{size}})
//     ...
//     --- End jit-post-opt ---
//     3: 39 39
//     yk-jit-event: enter-jit-code
//     2: 39 39
//     1: 39 39
//     yk-jit-event: deoptimise

// Check that idempotent functions work.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_idempotent))
uintptr_t add(uintptr_t x, uintptr_t y) {
  return x + y;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t i = 4;
  uintptr_t j = 37;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(j);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    // This call to the idempotent function cannot be elided as the trace
    // optimiser will be unable to figure out that `j` is constant.
    uintptr_t l = add(j, 2);
    uintptr_t k = yk_promote(j);
    // This call to the idempotent function will be elided.
    uintptr_t m = add(k, 2);
    fprintf(stderr, "%zu: %" PRIuPTR " %" PRIuPTR "\n", i, l, m);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
