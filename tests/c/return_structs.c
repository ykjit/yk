// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4:
//     1
//     999
//     2
//     77
//     888
//     77
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
//     %{{18_2}}: {0: i64, 64: i64} = icall %...
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{call}}: i128 = call %{{_}}() ; @make_struct
//     %{{s1_a}}: i8 = extractval %{{call}} [0]
//     %{{s1_b}}: i64 = extractval %{{call}} [64]
//     ...
//     %{{call2}}: i128 = call %{{_}}() ; @make_struct2
//     %{{s2_a}}: i64 = extractval %{{call2}} [0]
//     %{{s2_bc}}: i64 = extractval %{{call2}} [64]
//     ...
//     %{{s2_b}}: i8 = load %{{_}}
//     ...
//     %{{s2_c}}: i32 = load %{{_}}
//     ...
//     %{{50}}: ptr = load %5
//     %{{51}}: i128 = call %{{50}}()
//     %{{52}}: i64 = extractval %{{51}} [0]
//     %{{53}}: i64 = extractval %{{51}} [64]
//     ...
//     --- End hir ---
//     3:
//     1
//     999
//     2
//     77
//     888
//     77
//     yk-execution: enter-jit-code {"trid": "0"}
//     2:
//     1
//     999
//     2
//     77
//     888
//     77
//     1:
//     1
//     999
//     2
//     77
//     888
//     77
//     yk-execution: deoptimise ...
//     exit

// Check that `extractval`s are generated correctly for function return types.

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
  struct S2 (*p)(void) = make_struct2;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  NOOPT_VAL(p);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    struct S s1 = make_struct();
    fprintf(stderr, "%d:\n", i);
    fprintf(stderr, "%d\n", s1.a);
    fprintf(stderr, "%ld\n", s1.b);
    struct S2 s2 = make_struct2();
    fprintf(stderr, "%ld\n", s2.a);
    fprintf(stderr, "%d\n", s2.b);
    fprintf(stderr, "%u\n", s2.c);
    fprintf(stderr, "%u\n", p().b);
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
