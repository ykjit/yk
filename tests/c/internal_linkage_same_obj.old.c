// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     jitstate: enter-jit-code
//     ...

// Check that we can call a static function with internal linkage from the same
// compilation unit.

#include <assert.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) static int call_me(int x) {
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

  int res = 0, i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    res = call_me(argc);
    i--;
  }
  assert(res == 0);

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
