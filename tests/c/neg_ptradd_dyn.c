// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=9
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{15_4}}: ptr = ptr_add %{{15_1}}, 0 + (%{{15_3}} * {{4}})
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: ptr = dynptradd %{{_}}, %{{_}}, {{4}}
//     ...
//     --- End jit-pre-opt ---
//     i=9
//     yk-execution: enter-jit-code
//     i=9
//     i=9
//     yk-execution: deoptimise ...

// Check that basic trace compilation works.

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

  int arr[300];
  for (int x = 0; x < 300; x++) {
    arr[x] = x;
  }

  int i = 0;
  int *ptr = &arr[10];
  int minus1 = -1;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i < 4) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(ptr);
    NOOPT_VAL(minus1);
    fprintf(stderr, "i=%d\n", ptr[minus1]);
    i++;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
