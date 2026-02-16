// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{12}}: i32 = call %{{_}}(%{{_}}) ; @call_me
//     ...
//     --- End jit-pre-opt ---

// Check that a loopy function annotated `yk_unroll_safe` always gets inlined
// into the trace.
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

// A function containing a loop and marked `yk_unroll_safe`.
//
// We mark is `noinline` as well because we want to test that it gets inlined
// during tracing, not during AOT compilation.
__attribute__((yk_unroll_safe, noinline)) void inline_into_trace(int x) {
  while (x--)
    call_me(x);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 7;
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
