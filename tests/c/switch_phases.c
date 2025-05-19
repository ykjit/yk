// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i1 = eq %{{_}}, 0i32
//     guard true, %{{1}}, ...
//     ...
//     --- End jit-pre-opt ---
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{2}}: i1 = eq %{{_}}, 1i32
//     guard true, %{{2}}, ...
//     ...
//     --- End jit-pre-opt ---
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{3}}: i1 = eq %{{_}}, 2i32
//     guard true, %{{3}}, ...
//     ...
//     --- End jit-pre-opt ---
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{4}}: i1 = eq %{{c}}, 0i32
//     %{{5}}: i1 = eq %{{c}}, 1i32
//     %{{6}}: i1 = eq %{{c}}, 2i32
//     %{{7}}: i1 = or %{{4}}, %{{5}}
//     %{{8}}: i1 = or %{{7}}, %{{6}}
//     guard false, %{{8}}, ...
//     ...
//     --- End jit-pre-opt ---
//     ...

// Check that guards for switches are emitted correctly, including in
// side-traces.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

// This basically splits the loop in main() into four phases.
__attribute__((yk_outline))
int getj(int i) {
  int j;
  if (i < 10) {
    j = 0;
  }else if ((i >= 10) && (i < 20)) {
    j = 1;
  }else if ((i >= 20) && (i < 30)) {
    j = 2;
  } else {
    j = 99;
  }
  NOOPT_VAL(j);
  return j;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  NOOPT_VAL(loc);
  int i = 0;
  NOOPT_VAL(i);
  while (i < 40) {
    yk_mt_control_point(mt, &loc);

    int j = getj(i);
    switch (j) {
      case 0:
        fprintf(stderr, "%d %d: zero\n", i, j);
        break;
      case 1:
        fprintf(stderr, "%d %d: one\n", i, j);
        break;
      case 2:
        fprintf(stderr, "%d %d: two\n", i, j);
        break;
      default:
        fprintf(stderr, "%d %d: default\n", i, j);
        break;
    }
    i++;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
