// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     eq: 1
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     *%{{_}} = const_ptrtoint(@g)
//     ...
//     %{{_}}: i1 = eq %{{_}}, const_ptrtoint(@g)
//     ...
//     --- End aot ---
//     eq: 1
//     yk-execution: enter-jit-code {"trid": "0"}
//     eq: 1
//     eq: 1
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that a constant `ptrtoint` expression (the address of a global cast to an integer)
// is handled.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

// Must be a global
int g;

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    long addr = (long)&g;
    fprintf(stderr, "eq: %d\n", addr == (long)&g);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
