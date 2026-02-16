// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 41 41
//     4: 43 43
//     yk-tracing: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_idempotent, yk_outline]
//     func add_uint32_t(...
//     ...
//     #[yk_idempotent, yk_outline]
//     func add_uint64_t(...
//     ...
//     func main(...
//     ...
//     %{{_}}: i64 = call idempotent add_uint64_t(...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{6}}: i32 = 3
//     ...
//     %{{8}}: i32 = call %{{_}}(%{{_}}, %{{6}}) ; @__yk_opt_add_uint32_t
//     ...
//     %{{10}}: i64 = 4
//     ...
//     %{{12}}: i64 = call %{{_}}(%{{_}}, %{{10}}) ; @__yk_opt_add_uint64_t
//     ...
//     %{{21}}: i32 = 41
//     ...
//     %{{23}}: i64 = 43
//     ...
//     %{{26}}: i64 = load %{{_}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{26}}, %{{8}}, %{{21}}) ; @fprintf
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{12}}, %{{23}}) ; @fprintf
//     ...
//     --- End jit-pre-opt ---
//     3: 41 41
//     3: 43 43
//     yk-execution: enter-jit-code
//     2: 41 41
//     2: 43 43
//     1: 41 41
//     1: 43 43
//     yk-execution: deoptimise ...

// Check that idempotent functions work, both when arguments are and aren't promoted.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_idempotent))
uint32_t add_uint32_t(uint32_t x, uint32_t y) {
  return x + y;
}

__attribute__((yk_idempotent))
uint64_t add_uint64_t(uint64_t x, uint64_t y) {
  return x + y;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t li = 4;
  uint32_t k = 38;
  uint64_t l = 39;
  NOOPT_VAL(loc);
  NOOPT_VAL(li);
  NOOPT_VAL(k);
  NOOPT_VAL(l);
  while (li > 0) {
    yk_mt_control_point(mt, &loc);
    // This call to the idempotent function cannot be elided as the trace
    // optimiser will be unable to figure out that `k` & `l` are constants.
    uint32_t b = add_uint32_t(k, 3);
    uint64_t c = add_uint64_t(l, 4);
    uint32_t e = yk_promote(k);
    uint64_t f = yk_promote(l);
    // These calls to idempotent functions will be elided.
    uint32_t h = add_uint32_t(e, 3);
    uint64_t i = add_uint64_t(f, 4);
    fprintf(stderr, "%zu: %" PRIu32 " %" PRIu32 "\n", li, b, h);
    fprintf(stderr, "%zu: %" PRIu64 " %" PRIu64 "\n", li, c, i);
    li--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
