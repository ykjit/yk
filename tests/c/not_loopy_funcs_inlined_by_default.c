// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{12}}: i32 = call %{{_}}(%{{6}}) ; @call_me
//     ...
//     --- End jit-pre-opt ---

// Check that functions containing no loops get inlined into the trace.
//
// We can only see a call to `call_me()` in the trace if `inline_into_trace()`
// itself was also inlined into the trace.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

int call_me(int x); // from extra_linkage/call_me.c

// A function with no loops.
__attribute__((noinline)) void inline_into_trace(int x) { call_me(x); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    inline_into_trace(i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
