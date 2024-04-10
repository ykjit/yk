// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   stderr:
//     --- Begin aot ---
//     ...
//     call void @never_inline_into_trace(i32 noundef ...
//     ...
//     --- End aot ---
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(ptr %0, ptr %1) {
//       ...
//       call void @never_inline_into_trace(i32 noundef ...
//       ...
//     }
//     ...
//     --- End jit-pre-opt ---

// Check a loopy function that is NOT annotated `yk_unroll_safe`:
//   - never gets linlined during AOT compilation.
//   - never gets inlined into the trace.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

int call_me(int x); // from extra_linkage/call_me.c

// A function containing a loop, not marked `yk_unroll_safe`.
void never_inline_into_trace(int x) {
  while (x--)
    call_me(x);
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
    never_inline_into_trace(i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
