// Run-time:
//   env-var: YKD_OPT=0
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stdout:
//     10
//     9
//     8
//     7
//     7: 6
//     7: 5
//     7: 4
//     7: 4: 3
//     7: 4: 2
//     7: 4: 1
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
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  int i = 10;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (i < 7)
      printf("7: ");
    if (i < 4)
      printf("4: ");
    printf("%d\n", i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  printf("exit\n");
  return (EXIT_SUCCESS);
}
