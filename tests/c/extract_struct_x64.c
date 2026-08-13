// ignore-if: test ${YK_ARCH} != "x86_64"
// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     1
//     999
//     2
//     77
//     888
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{10_1}}: {0: i8, 64: i64} = call make_struct()...
//     ...
//     %{{11_2}}: i8 = extractvalue %{{10_1}}, [0]
//     ...
//     %{{11_5}}: i64 = extractvalue %{{10_1}}, [1]
//     ...
//     %{{13_1}}: {0: i64, 64: i64} = call make_struct2()...
//     ...
//     %{{14_2}}: i64 = extractvalue %{{13_1}}, [0]
//     ...
//     %{{14_5}}: i64 = extractvalue %{{13_1}}, [1]
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{call}}: i64 = call %{{_}}() ; @make_struct
//     %{{s1_a}}: i8 = extractvalue %{{call}} [0]
//     ...
//     %{{s1_b}}: i64 = extractvalue %{{call}} [64]
//     ...
//     %{{call2}}: i64 = call %{{_}}() ; @make_struct2
//     %{{s2_a}}: i64 = extractvalue %{{call2}} [0]
//     ...
//     %{{s2_bc}}: i64 = extractvalue %{{call2}} [64]
//     ...
//     %{{s2_b}}: i8 = load %{{_}}
//     ...
//     %{{s2_c}}: i32 = load %{{_}}
//     ...
//     --- End hir ---
//     1
//     999
//     2
//     77
//     888
//     yk-execution: enter-jit-code {"trid": "0"}
//     1
//     999
//     2
//     77
//     888
//     1
//     999
//     2
//     77
//     888
//     yk-execution: deoptimise ...
//     exit

// Test extractvalue non-inlined calls returning structs.
// The expected AOT/HIR output below assumes the x64 sysvabi register-packing for
// small struct returns - two 64-bit GP registers.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

struct S {
  uint8_t a;
  uint64_t b;
};

__attribute__((yk_outline))
struct S make_struct() {
  struct S ret = {1, 999};
  return ret;
}

struct S2 {
  uint64_t a;
  uint8_t b;
  uint32_t c;
};

__attribute__((yk_outline))
struct S2 make_struct2() {
  struct S2 ret = {2, 77, 888};
  return ret;
}

void interp(){
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    struct S s1 = make_struct();
    fprintf(stderr, "%d\n", s1.a);
    fprintf(stderr, "%ld\n", s1.b);
    struct S2 s2 = make_struct2();
    fprintf(stderr, "%ld\n", s2.a);
    fprintf(stderr, "%d\n", s2.b);
    fprintf(stderr, "%u\n", s2.c);
    i--;
  }
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
}

int main(int argc, char **argv) {
  interp();
  return (EXIT_SUCCESS);
}
