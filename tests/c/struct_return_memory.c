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
//     40000000000
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     call make_triple(%{{buf}})...
//     ...
//     %{{f0}}: ptr = ptr_add %{{buf}}, 0
//     %{{_}}: i8 = load %{{f0}}
//     ...
//     %{{f1}}: ptr = ptr_add %{{buf}}, 2
//     %{{_}}: i16 = load %{{f1}}
//     ...
//     %{{f2}}: ptr = ptr_add %{{buf}}, 4
//     %{{_}}: i32 = load %{{f2}}
//     ...
//     %{{f3}}: ptr = ptr_add %{{buf}}, 8
//     %{{_}}: i64 = load %{{f3}}
//     ...
//     %{{f4}}: ptr = ptr_add %{{buf}}, 16
//     %{{_}}: i64 = load %{{f4}}
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     --- End hir ---
//     10
//     200
//     30000
//     20000000000
//     40000000000
//     yk-execution: enter-jit-code {"trid": "0"}
//     10
//     200
//     30000
//     20000000000
//     40000000000
//     10
//     200
//     30000
//     20000000000
//     40000000000
//     yk-execution: deoptimise ...
//     ...

// Check that a struct too large to fit in two registers (over 16 bytes) is
// returned via a pointer argument rather than packed into registers, and that
// reading its fields back out compiles and executes correctly.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

struct Triple {
  uint8_t small;
  uint16_t mid;
  uint32_t big32;
  uint64_t big;
  uint64_t big2;
};

struct Triple make_triple() {
  struct Triple t = {10, 200, 30000, 20000000000ULL, 40000000000ULL};
  return t;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    struct Triple t = make_triple();
    fprintf(stderr, "%u\n", t.small);
    fprintf(stderr, "%u\n", t.mid);
    fprintf(stderr, "%u\n", t.big32);
    fprintf(stderr, "%llu\n", (unsigned long long)t.big);
    fprintf(stderr, "%llu\n", (unsigned long long)t.big2);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
