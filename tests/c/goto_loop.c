// Run-time:
//   stdout: 1000
//   stderr:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 0;
  NOOPT_VAL(i);
loop:
  yk_mt_control_point(mt, &loc);
  if (i < 1000) {
    i++;
    goto loop;
  }
  printf("%d", i);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
