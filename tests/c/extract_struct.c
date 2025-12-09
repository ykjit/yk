// ignore-if: test "$YK_JITC" = "j2" # not yet implemented in j2
// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     2
//     999
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{10_1}}: {0: i8, 64: i64} = call make_struct()...
//     ...
//     %{{11_2}}: i8 = extractvalue %{{10_1}}, [0]
//     ...
//     %{{11_5}}: i64 = extractvalue %{{10_1}}, [1]
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2
//     999
//     yk-execution: enter-jit-code
//     2
//     999
//     2
//     999
//     yk-execution: deoptimise ...
//     exit

// Test that trace builder handles loading structs by rewriting `extractvalue`
// instructions into `ptr_add`s and `load`s. We only check that the AOT IR
// contains a struct load and check that the trace does the correct thing.
// Matching on the JIT IR is difficult since the `ptr_add`s and `load`s are
// not easily distinguished from unrelated `ptr_add`s and `load`s.

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

struct S make_struct() {
  struct S ret = {1, 999};
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
    s1.a = 2;
    fprintf(stderr, "%d\n", s1.a);
    fprintf(stderr, "%ld\n", s1.b);
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
