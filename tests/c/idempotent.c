// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4: 39 39
//     4: 41 41
//     4: 43 43
//     yk-jit-event: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_idempotent, yk_outline]
//     func add_uintptr_t(...
//     ...
//     #[yk_idempotent, yk_outline]
//     func add_uint32_t(...
//     ...
//     #[yk_idempotent, yk_outline]
//     func add_uint64_t(...
//     ...
//     func main(...
//     ...
//     %{{_}}: i{{size}} = idempotent_promote %{{_}}
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i{{size}} = call @add_uintptr_t(%{{_}}, 2i{{size}}) <idem_const 39i{{size}}>
//     ...
//     %{{_}}: i32 = call @add_uint32_t(%{{_}}, 3i32) <idem_const 41i32>
//     ...
//     %{{_}}: i64 = call @add_uint64_t(%{{_}}, 4i64) <idem_const 43i64>
//     ...
//     %{{_}}: i{{size}} = call @add_uintptr_t(%{{_}}, 2i{{size}}) <idem_const 39i{{size}}>
//     ...
//     %{{_}}: i32 = call @add_uint32_t(%{{_}}, 3i32) <idem_const 41i32>
//     ...
//     %{{_}}: i64 = call @add_uint64_t(%{{_}}, 4i64) <idem_const 43i64>
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     %{{_}}: i{{size}} = call @add_uintptr_t(%{{_}}, 2i{{size}}) <idem_const 39i{{size}}>
//     ...
//     %{{_}}: i32 = call @add_uint32_t(%{{_}}, 3i32) <idem_const 41i32>
//     ...
//     %{{_}}: i64 = call @add_uint64_t(%{{_}}, 4i64) <idem_const 43i64>
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, %{{_}}, 39i{{size}})
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, %{{_}}, 41i32)
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, %{{_}}, 43i64)
//     ...
//     --- End jit-post-opt ---
//     3: 39 39
//     3: 41 41
//     3: 43 43
//     yk-jit-event: enter-jit-code
//     2: 39 39
//     2: 41 41
//     2: 43 43
//     1: 39 39
//     1: 41 41
//     1: 43 43
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
uintptr_t add_uintptr_t(uintptr_t x, uintptr_t y) {
  return x + y;
}

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
  uintptr_t j = 37;
  uint32_t k = 38;
  uint64_t l = 39;
  NOOPT_VAL(loc);
  NOOPT_VAL(li);
  NOOPT_VAL(j);
  NOOPT_VAL(k);
  NOOPT_VAL(l);
  while (li > 0) {
    yk_mt_control_point(mt, &loc);
    // This call to the idempotent function cannot be elided as the trace
    // optimiser will be unable to figure out that `j` is constant.
    uintptr_t a = add_uintptr_t(j, 2);
    uint32_t b = add_uint32_t(k, 3);
    uint64_t c = add_uint64_t(l, 4);
    uintptr_t d = yk_promote(j);
    uint32_t e = yk_promote(k);
    uint64_t f = yk_promote(l);
    // These calls to idempotent functions will be elided.
    uintptr_t g = add_uintptr_t(d, 2);
    uint32_t h = add_uint32_t(e, 3);
    uint64_t i = add_uint64_t(f, 4);
    fprintf(stderr, "%zu: %" PRIuPTR " %" PRIuPTR "\n", li, a, g);
    fprintf(stderr, "%zu: %" PRIu32 " %" PRIu32 "\n", li, b, h);
    fprintf(stderr, "%zu: %" PRIu64 " %" PRIu64 "\n", li, c, i);
    li--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
