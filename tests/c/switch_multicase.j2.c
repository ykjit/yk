// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
//   stderr:
//     yk-tracing: start-tracing
//     0: 0
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{4}}: i32 = 0
//     %{{5}}: i1 = icmp eq %{{3}}, %{{4}}
//     %{{6}}: i32 = 1
//     %{{7}}: i1 = icmp eq %{{3}}, %{{6}}
//     %{{8}}: i1 = or %{{5}}, %{{7}}
//     %{{9}}: i32 = 2
//     %{{10}}: i1 = icmp eq %{{3}}, %{{9}}
//     %{{11}}: i1 = or %{{8}}, %{{10}}
//     %{{12}}: i32 = 3
//     %{{13}}: i1 = icmp eq %{{3}}, %{{12}}
//     %{{14}}: i1 = or %{{11}}, %{{13}}
//     %{{15}}: i32 = 4
//     %{{16}}: i1 = icmp eq %{{3}}, %{{15}}
//     %{{17}}: i1 = or %{{14}}, %{{16}}
//     guard true, %17, ...
//     ...
//     --- End jit-pre-opt ---
//     1: 0
//     2: 0
//     3: 0
//     4: 0
//     yk-tracing: start-side-tracing
//     5: 1
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     6: 1
//     7: 1
//     8: 1
//     yk-tracing: start-side-tracing
//     exit

// Check that guards for switches are emitted correctly, including in
// side-traces.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  NOOPT_VAL(loc);
  int i = 0;
  NOOPT_VAL(i);
  while (i < 9) {
    yk_mt_control_point(mt, &loc);

    switch (i) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
        fprintf(stderr, "%d: 0\n", i);
        break;
      default:
        fprintf(stderr, "%d: 1\n", i);
        break;
    }
    i++;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
