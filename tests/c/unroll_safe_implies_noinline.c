// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:aot
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     call never_aot_inline(...
//     ...
//     --- End aot ---
//     ...

// Check that `yk_unroll_safe` implies `noinline` (thus blocking AOT inlining).

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

int call_me(int x); // from extra_linkage/call_me.c

// A function containing a loop and marked `yk_unroll_safe`.
__attribute__((yk_unroll_safe)) void never_aot_inline(int x) {
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
    never_aot_inline(i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
