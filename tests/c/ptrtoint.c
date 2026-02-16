// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     ptr: {{ptr}}
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{4}}: i64 = ptrtoint %{{1}}
//     ...
//     --- End jit-pre-opt ---
//     ptr: {{ptr}}
//     yk-execution: enter-jit-code
//     ptr: {{ptr}}
//     ptr: {{ptr}}
//     yk-execution: deoptimise ...
//     exit

// Check that pointer to integer conversion works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    long ptr = (long)&loc;
    fprintf(stderr, "ptr: %ld\n", ptr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
