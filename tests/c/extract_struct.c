// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     s1.a: 4, s2.a: 5
//     s1.b: 999, s2.b: 999
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{10_1}}: {0: i8, 64: i64} = call make_struct()...
//     ...
//     %{{11_2}}: i8 = extractvalue %{{10_1}}, [0]
//     ...
//     %{{11_5}}: i64 = extractvalue %{{10_1}}, [1]
//     ...
//     %{{20_1}}: {0: i8, 64: i64} = call make_struct_inlined()...
//     ...
//     %{{21_2}}: i8 = extractvalue %{{20_1}}, [0]
//     ...
//     %{{21_5}}: i64 = extractvalue %{{20_1}}, [1]
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     --- End hir ---
//     s1.a: 3, s2.a: 4
//     s1.b: 999, s2.b: 999
//     yk-execution: enter-jit-code {"trid": "0"}
//     s1.a: 2, s2.a: 3
//     s1.b: 999, s2.b: 999
//     s1.a: 1, s2.a: 2
//     s1.b: 999, s2.b: 999
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Test that `extractvalue` on a call struct return works for both representations the trace 
// builder can produce: 
// 1. register-packed (RAX:RDX, from a real non-inlined `call`)
// 2. memory-backed (`ptr_add` + `load`, from an inlined callee's own struct-typed `load`. 
// We only check that the AOT IR contains the expected `call`/`extractvalue`
// shape and that the trace does the correct thing at runtime; matching on the JIT IR itself is
// difficult since its `ptr_add`s and `load`s aren't easily distinguished from unrelated ones.

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

// `yk_outline` stops the trace-builder inlining this call, so it stays a real, non-inlined
// `call` in the trace. That's a genuine machine-call boundary, so the SysV ABI packs this
// struct's return into RAX:RDX.
__attribute__((yk_outline))
struct S make_struct() {
  struct S ret = {1, 999};
  return ret;
}

// This call is inlined, so struct return value is always set as real memory.
struct S make_struct_inlined() {
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
    s1.a = i;
    struct S s2 = make_struct_inlined();
    s2.a = i+1;
    fprintf(stderr, "s1.a: %d, s2.a: %d\n", s1.a, s2.a);
    fprintf(stderr, "s1.b: %ld, s2.b: %ld\n", s1.b, s2.b);
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
