// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     func f(%arg0: i32) -> i32 {
//     ...
//     # extract_struct_local.c:{{_}}: return x.f;
//     %{{s_addr}}: ptr = ptr_add %{{_}}, 0
//     %{{s_val}}: i32 = load %{{s_addr}}
//     ...
//     ret %{{s_val}}
//     }
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{f_result}}: i32 = load %{{_}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{f_result}}) ; @fprintf
//     ...
//     --- End hir ---
//     3
//     2
//     1
//     exit

// Test that returning a field from an entirely local struct to a function
// is read via a plain pointer load and not as an extractval instruction.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

struct S {
  int f;
};

int f(int v) {
  struct S x = {v};
  return x.f;
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
    fprintf(stderr, "%d\n", f(i));
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
