// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 41 41
//     4: 43 43
//     4: 98 98
//     yk-tracing: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_idempotent, yk_outline, memory(none)]
//     func add_uint32_t(...
//     ...
//     #[yk_idempotent, yk_outline, memory(none)]
//     func add_uint64_t(...
//     ...
//     #[yk_idempotent, yk_outline, memory(none)]
//     func next_ptr(...
//     ...
//     #[yk_idempotent, yk_outline, memory(read)]
//     func byte_at(...
//     ...
//     func main(...
//     ...
//     %{{_}}: i64 = call idempotent add_uint64_t(...
//     ...
//     --- End aot ---
//     --- Begin hir ---
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
//     %{{39}}: i8 = call %{{_}}(%{{_}}) ; @__yk_opt_byte_at
//     ...
//     %{{42}}: i32 = zext %{{39}}
//     ...
//     %{{43}}: i32 = 98
//     ...
//     %{{44}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{42}}, %{{43}}) ; @fprintf
//     ...
//     --- End hir ---
//     3: 41 41
//     3: 43 43
//     3: 98 98
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 41 41
//     2: 43 43
//     2: 98 98
//     1: 41 41
//     1: 43 43
//     1: 98 98
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}

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

__attribute__((yk_idempotent))
uint8_t *next_ptr(uint8_t *x) {
  // Calling `add_uint64_t` is a hack that, at least for now, stops LLVM from
  // inlining this function.
  return x + add_uint64_t(0, 1);
}

__attribute__((yk_idempotent))
uint8_t byte_at(uint8_t *x) {
  return *x;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  uint8_t *bytes = (uint8_t *) "abcd";

  size_t li = 4;
  uint32_t k = 38;
  uint64_t l = 39;
  uint8_t *m = bytes;
  NOOPT_VAL(loc);
  NOOPT_VAL(li);
  NOOPT_VAL(k);
  NOOPT_VAL(l);
  NOOPT_VAL(m);
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

    // Now we do the same thing for pointers: first things that can't be
    // elided.
    uint8_t *p1 = next_ptr(m);
    uint8_t *p2 = yk_promote(m);
    uint8_t pc1 = byte_at(p1);
    // Now things that can be elided.
    uint8_t *p3 = next_ptr(p2);
    uint8_t pc3 = byte_at(p3);
    fprintf(stderr, "%zu: %" PRIu8  " %" PRIu8  "\n", li, pc1, pc3);
    li--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
