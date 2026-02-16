// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 5
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{8}}: ptr = 0x{{_}} ; @callback
//     ...
//     %{{10}}: i32 = call %{{_}}(%{{8}}, %{{_}}, %{{_}}) ; @call_callback
//     ...
//     --- End jit-pre-opt ---
//     ...
//     3: 4
//     yk-execution: enter-jit-code
//     2: 3
//     1: 2
//     yk-execution: deoptimise ...


// Check that we can reliably deal with "foreign" (not compiled with ykllvm)
// code that calls back into "native code".

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int call_callback(int (*callback)(int, int), int x, int y);

__attribute((noinline)) int callback(int x, int y) { return (x + y) / 2 + 1; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %d\n", i,  call_callback(&callback, i, i));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
