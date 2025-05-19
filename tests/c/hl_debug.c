// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     yk-tracing: start-tracing: somefile.lua:1234: for i = 0, 10 do
//     4
//     3
//     yk-tracing: stop-tracing: somefile.lua:1234: for i = 0, 10 do
//     2
//     yk-tracing: start-tracing: someotherfile.lua:5678: while j < 1000
//     1
//     exit

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

  YkLocation loc1 = yk_location_new();
  yk_location_set_debug_str(&loc1, "somefile.lua:1234: for i = 0, 10 do");

  YkLocation loc2 = yk_location_new();
  yk_location_set_debug_str(&loc2, "someotherfile.lua:5678: while j < 1000");

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    YkLocation *loc = i % 2 == 0 ? &loc1 : &loc2;
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
