// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   stderr:
//     yk-jit-event: start-tracing
//     x=0
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{12}}: i32 = call @call_me(%{{8}})
//     ...
//     --- End jit-pre-opt ---
//     ...
//     yk-jit-event: enter-jit-code
//     x=0
//     x=0
//     yk-jit-event: deoptimise

// Check that we can call a static function with internal linkage from the same
// compilation unit.

#include <assert.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline, yk_outline)) static int call_me(int x) {
  NOOPT_VAL(x);
  if (x == 5)
    return 1;
  else
    return 0;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int x = 999, i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    x = call_me(argc);
    fprintf(stderr, "x=%d\n", x);
    i--;
  }
  assert(x == 0);

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
