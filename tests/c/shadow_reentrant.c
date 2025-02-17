// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     i=4, x=0, argc=1
//     i=3, x=0, argc=1
//     i=2, x=0, argc=1
//     i=1, x=0, argc=1

// Check that the shadow stack is properly allocated when we "call-back" from
// external code.
//
// An earlier bug meant that we didn't allocate shadow stack for calls to
// external code, meaning that when we called back we trashed the shadow stack
// of the previous frame. In this test that manifested as argc being mutated!

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int call_callback(int (*callback)(int, int), int x, int y);

__attribute((noinline)) int callback(int x, int y) { return (x + y) / 2; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 100);
  YkLocation loc = yk_location_new();

  int x = 0;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(x);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d, x=%d, argc=%d\n", i, x, argc);
    call_callback(&callback, i, i);
    i--;
  }
  NOOPT_VAL(x);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
