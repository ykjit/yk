// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4: 28
//     yk-jit-event: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_idempotent, yk_outline]
//     func idem2(...
//     ...
//     #[yk_idempotent, yk_outline]
//     func idem1(...
//     ...
//     func main(...
//     ...
//     %{{_}}: i{{size}} = idempotent_promote %{{_}}
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i{{size}} = call @idem1(%{{_}}, %{{_}}) <idem_const 10i{{size}}>
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     %31: i32 = call @fprintf(%{{_}}, %{{_}}, 4i{{size}}, 10i{{size}})
//     ...
//     --- End jit-post-opt ---
//     3: 24
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     2: 20
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     1: 16

// Check that idempotent functions work when an idempotent function itself
// calls more idempotent functions. This forces the trace builder to outline
// through the outer function and thus discard idemconsts from the inner
// functions.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_idempotent))
uintptr_t idem2(uintptr_t x, uintptr_t y) {
  return x + y;
}

__attribute__((yk_idempotent))
uintptr_t idem1(uintptr_t x, uintptr_t y) {
  // idemconsts made here will be consumed and discarded by the trace builder.
  uintptr_t a = idem2(x + 1, y + 1);
  uintptr_t b = idem2(y + 5, x + 5);
  return a + b;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    uintptr_t k = yk_promote(i);
    fprintf(stderr, "%" PRIuPTR ": %" PRIuPTR "\n", i, idem1(k, k));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
