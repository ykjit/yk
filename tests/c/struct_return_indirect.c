// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=hir,aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-tracing: start-tracing
//     10
//     200
//     30000
//     20000000000
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{9_1}}: {0: i64, 64: i64} = icall %{{_}}()...
//     ...
//     %{{_}}: i64 = extractvalue %{{9_1}}, [0]
//     ...
//     %{{_}}: i64 = extractvalue %{{9_1}}, [1]
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     --- End hir ---
//     10
//     200
//     30000
//     20000000000
//     yk-execution: enter-jit-code {"trid": "0"}
//     10
//     200
//     30000
//     20000000000
//     10
//     200
//     30000
//     20000000000
//     yk-execution: deoptimise ...
//     ...

// Check that a struct returned by value from an indirect call compiles and executes correctly.
// Here the make_pair call is never inlined, so the struct genuinely comes back packed 
// in `RAX` and `RDX`.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

struct Pair {
  uint8_t small;
  uint16_t mid;
  uint32_t big32;
  uint64_t big;
};

__attribute__((noinline)) struct Pair make_pair() {
  struct Pair p = {10, 200, 30000, 20000000000ULL};
  return p;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  struct Pair (*fn)() = make_pair;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(fn);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    struct Pair p = fn();
    fprintf(stderr, "%u\n", p.small);
    fprintf(stderr, "%u\n", p.mid);
    fprintf(stderr, "%u\n", p.big32);
    fprintf(stderr, "%llu\n", (unsigned long long)p.big);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
