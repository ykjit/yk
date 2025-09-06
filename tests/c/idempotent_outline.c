// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 28
//     yk-tracing: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_idempotent, yk_outline]
//     func f(...
//     ...
//     func main(...
//     ...
//     %{{_}}: i{{size}} = idempotent_promote %{{_}}
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i{{size}} = call @{{__yk_unopt_:f}}(%{{v}}, %{{v}}) <idem_const 28i{{size}}>
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, 4i{{size}}, 28i{{size}})
//     ...
//     --- End jit-post-opt ---
//     3: 24
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     2: 20
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     1: 16

// Check that idempotent functions work when they call functions that
// themselves promote values. This forces the trace builder to outline over the
// inner promotions, discarding their associated idemconsts.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

uintptr_t h(uintptr_t x, uintptr_t y) {
  return x + y;
}

__attribute__((noinline))
uintptr_t g(uintptr_t x, uintptr_t y) {
  // idemconsts made here will be consumed and discarded by the trace builder.
  uintptr_t a = yk_promote(h(x + 1, y + 1));
  uintptr_t b = yk_promote(h(y + 5, x + 5));
  return a + b;
}

__attribute__((yk_idempotent))
uintptr_t f(uintptr_t x, uintptr_t y) {
  return g(x, y);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  // Prevent the OutlineUntraceable pass from marking g() yk_outline.
  g(0, 0);

  size_t i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    uintptr_t k = yk_promote(i);
    fprintf(stderr, "%" PRIuPTR ": %" PRIuPTR "\n", i, f(k, k));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
