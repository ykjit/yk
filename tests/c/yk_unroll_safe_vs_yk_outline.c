// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     #[yk_outline]
//     func never_inline_into_trace(...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     call %{{_}}(%{{_}}) ; @__yk_opt_never_inline_into_trace
//     ...
//     --- End jit-pre-opt ---

// Check that `yk_outline` wins over `yk_unroll_safe`.
//
// Although it is tempting to have clang emit an error when these conflicting attributes are used
// together, the idiomatic "clang way" is to have one attribute win out over the other.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

int call_me(int x); // from extra_linkage/call_me.c

// Both `yk_outline` and `yk_unroll_safe`!
__attribute__((yk_outline, yk_unroll_safe)) void
never_inline_into_trace(int x) {
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
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
