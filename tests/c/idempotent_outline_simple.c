// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4: {{val}}
//     yk-jit-event: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     func g(...
//     ...
//     %{{1_0}}: i64 = promote %{{_}} ...
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
//     %{{_}}: i{{size}} = call @f(1i{{size}}) <idem_const 2i{{size}}>
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, 2i{{size}})
//     ...
//     --- End jit-post-opt ---
//     ...

// Check that idempotent functions work when they call functions that
// themselves promote values. This forces the trace builder to outline over the
// inner promotions, discarding their associated idemconsts.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline))
uintptr_t g(uintptr_t x) {
  return yk_promote(x) + 1;
}

__attribute__((yk_idempotent))
uintptr_t f(uintptr_t x) {
  return g(x);
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
    fprintf(stderr, "%" PRIuPTR ": %" PRIuPTR "\n", i, f(1));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
