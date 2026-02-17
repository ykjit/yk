// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
//   stderr:
//     yk-tracing: start-tracing
//     0 0: zero
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{9}}: i32 = 0
//     %{{10}}: i1 = icmp eq %{{_}}, %{{9}}
//     guard true, %{{10}}, ...
//     ...
//     --- End jit-pre-opt ---
//     1 0: zero
//     2 0: zero
//     3 0: zero
//     4 0: zero
//     5 0: zero
//     6 0: zero
//     7 0: zero
//     8 0: zero
//     9 0: zero
//     10 1: one
//     11 1: one
//     12 1: one
//     13 1: one
//     yk-tracing: start-side-tracing
//     14 1: one
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{5}}: i32 = 1
//     %{{6}}: i1 = icmp eq %{{_}}, %{{5}}
//     guard true, %{{6}}, ...
//     ...
//     --- End jit-pre-opt ---
//     15 1: one
//     16 1: one
//     17 1: one
//     18 1: one
//     19 1: one
//     20 2: two
//     21 2: two
//     22 2: two
//     23 2: two
//     yk-tracing: start-side-tracing
//     24 2: two
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{5}}: i32 = 2
//     %{{6}}: i1 = icmp eq %{{_}}, %{{5}}
//     guard true, %{{6}}, ...
//     ...
//     --- End jit-pre-opt ---
//     25 2: two
//     26 2: two
//     27 2: two
//     28 2: two
//     29 2: two
//     30 99: default
//     31 99: default
//     32 99: default
//     33 99: default
//     yk-tracing: start-side-tracing
//     34 99: default
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{4}}: i32 = arg
//     %{{5}}: ...
//     ...
//     --- End jit-pre-opt ---
//     35 99: default
//     36 99: default
//     37 99: default
//     38 99: default
//     39 99: default
//     exit

// Check that guards for switches are emitted correctly, including in
// side-traces, and that the `default` case does not lead to a guard being
// emitted.

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
